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
use quantize::{QuantTensor, tensor_from_f32_slice};
use sram::VectorSram;
use tch::Tensor;

use crate::runtime_config::{
    VECTOR_ADD_CYCLES, VECTOR_EXP_CYCLES, VECTOR_MAX_CYCLES, VECTOR_MIN_CYCLES, VECTOR_MUL_CYCLES,
    VECTOR_RECI_CYCLES, VECTOR_SOFTPLUS_CYCLES, VECTOR_SUM_CYCLES, VLEN,
};
use crate::{cycle, op};

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

    pub(crate) async fn mul_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() * (f as f64), a.data_type());
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
                    let updated = &sliced * (f as f64);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    /// `Vector[vd] += Vector[vs1] * f`.
    ///
    /// Mirrors `mul_scalar` but reads the destination too, so the accumulate is
    /// part of the instruction rather than a separate `V_ADD_VV` over a scratch
    /// row. The quantisation happens once, on the sum -- which is what makes it
    /// a *fused* multiply-add and not just a shorter encoding of the pair.
    pub(crate) async fn fma_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        let d = self.vram.read(vd).await;
        if rmask == 0 {
            let c =
                QuantTensor::quantize(d.as_tensor() + a.as_tensor() * (f as f64), d.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            // Masked-off heads keep the destination's existing value, so the
            // result starts as d and not as a -- the opposite of mul_scalar,
            // where the source is the base.
            let result = d.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let addend = a.as_tensor().narrow(0, start, end - start) * (f as f64);
                    let updated = &sliced + &addend;
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, d.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn max_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor().clamp_min(f as f64), a.data_type());
            cycle!(*VECTOR_MAX_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = sliced.clamp_min(f as f64);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MAX_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn min_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor().clamp_max(f as f64), a.data_type());
            cycle!(*VECTOR_MIN_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = sliced.clamp_max(f as f64);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MIN_CYCLES);
            self.vram.write(vd, c).await;
        }
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

    pub(crate) async fn sub(&self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
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

    pub(crate) async fn mul(&self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
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
            let result = clamped.shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced.exp();
                    result.narrow(0, start, end - start).copy_(updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_EXP_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    /// Elementwise `softplus(x) = log(1 + exp(x))`.
    ///
    /// Evaluated as `relu(x) + log1p(exp(-|x|))`. That identity is algebraically
    /// exact and never feeds `exp` a positive argument, so unlike the naive
    /// `log1p(exp(x))` it cannot overflow — no input clamp is needed and none is
    /// applied, which matters because Mamba's `dt` feeds `exp(A*dt)` and a clamp
    /// would silently flatten the decay for large `dt`.
    fn softplus_tensor(x: &Tensor) -> Tensor {
        x.clamp_min(0.0) + x.abs().neg().exp().log1p()
    }

    pub(crate) async fn softplus(&self, vd: u32, vs1: u32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(Self::softplus_tensor(a.as_tensor()), a.data_type());
            cycle!(*VECTOR_SOFTPLUS_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = Self::softplus_tensor(&sliced);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_SOFTPLUS_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    /// Read one whole VLEN-wide VRAM row out to the scalar domain.
    ///
    /// The inverse direction of [`Self::vector_transfer_fp`], and the engine behind
    /// `S_MAP_FP_V`. Charged the same VLEN cycles as the forward transfer.
    pub(crate) async fn vector_read_fp(&self, vs1: u32) -> Vec<bf16> {
        let a = self.vram.read(vs1).await;
        let values =
            Vec::<f32>::try_from(a.as_tensor()).expect("VRAM row must be convertible to Vec<f32>");
        cycle!(*VLEN);
        values.into_iter().map(bf16::from_f32).collect()
    }

    pub(crate) async fn reciprocal(&self, vd: u32, vs1: u32, rmask: u8, mask: u32) {
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
                    result.narrow(0, start, end - start).copy_(updated);
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
        let tensor = tensor_from_f32_slice(&f32_vec);
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
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced.sum(tch::Kind::Float);
                    result.narrow(0, start, end - start).copy_(updated);
                }
            }
            let val: f32 = result.sum(tch::Kind::Float).try_into().unwrap();
            f + val
        }
    }

    pub(crate) async fn reduce_max(&self, vs1: u32, f: f32, rmask: u8, mask: u32) -> f32 {
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
                    result.narrow(0, start, end - start).copy_(updated);
                }
            }
            let val: f32 = result.max().try_into().unwrap();
            f32::max(val, f)
        }
    }

    pub(crate) async fn topk_softmax(
        &self,
        vs1: u32,
        expert_count: usize,
        topk: usize,
    ) -> (Vec<u32>, Vec<bf16>) {
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
            let a = self.vram.read(vs1 + chunk_start as u32).await;
            let chunk_len = (expert_count - chunk_start).min(tile_size);
            for idx in 0..chunk_len {
                let value = a.as_tensor().double_value(&[idx as i64]) as f32;
                logits.push(if value.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    value
                });
            }
        }

        let mut ranked: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        ranked.sort_by(|(idx_a, val_a), (idx_b, val_b)| {
            val_b.total_cmp(val_a).then_with(|| idx_a.cmp(idx_b))
        });
        let selected = &ranked[..topk];

        let max_logit = selected
            .iter()
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_exp_values: Vec<f32> = selected
            .iter()
            .map(|(_, value)| (*value - max_logit).exp())
            .collect();
        let denom: f32 = selected_exp_values.iter().sum();
        // When every selected logit is NEG_INFINITY (whole row NaN/-inf), max_logit
        // is -inf, so `value - max_logit` is NaN, exp is NaN and denom is NaN — a
        // plain `denom == 0.0` check would miss it and emit NaN weights. Require a
        // finite, positive denominator; otherwise the weights are 0.0.
        let weights: Vec<bf16> = selected_exp_values
            .iter()
            .map(|value| {
                let w = if denom.is_finite() && denom > 0.0 {
                    value / denom
                } else {
                    0.0
                };
                bf16::from_f32(w)
            })
            .collect();
        let indices: Vec<u32> = selected.iter().map(|(idx, _)| *idx as u32).collect();

        cycle!((*VECTOR_MAX_CYCLES).saturating_mul(expert_count as u32));
        (indices, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{DataType, FpType, MxDataType};
    use runtime::{Executor, Instant};
    use std::sync::{Arc, Mutex};

    fn tensor_values(tensor: &Tensor) -> Vec<f32> {
        let len = tensor.size()[0] as usize;
        let data = unsafe { core::slice::from_raw_parts(tensor.data_ptr() as *const f32, len) };
        data.to_vec()
    }

    #[tokio::test]
    async fn fma_scalar_accumulates_into_the_destination() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(4, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 4, 2);
            let ty = MxDataType::Plain(fp_type);

            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]), ty),
            )
            .await;
            vram.write(
                4,
                QuantTensor::quantize(Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0]), ty),
            )
            .await;

            machine.fma_scalar(0, 4, 0.5, 0, u32::MAX).await;

            let out = vram.read(0).await;
            let src = vram.read(4).await;
            *got_task.lock().unwrap() = Some((
                tensor_values(out.as_tensor()),
                tensor_values(src.as_tensor()),
            ));
        });
        executor.enter(Instant::ETERNITY).await;

        let (dst, src) = got.lock().unwrap().take().unwrap();
        // d + a*f, not a*f: the accumulate is the instruction, not a side effect.
        assert_eq!(dst, vec![6.0, 12.0, 18.0, 24.0]);
        assert_eq!(src, vec![10.0, 20.0, 30.0, 40.0], "the source is read-only");
    }

    #[tokio::test]
    async fn fma_scalar_leaves_masked_off_heads_holding_the_destination() {
        // mul_scalar's masked path starts from the *source*; fma's must start
        // from the destination, or a masked-off lane silently loses whatever it
        // had accumulated so far.
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(4, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 4, 2);
            let ty = MxDataType::Plain(fp_type);

            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]), ty),
            )
            .await;
            vram.write(
                4,
                QuantTensor::quantize(Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0]), ty),
            )
            .await;

            // mask_unit 2, tile 4 -> two heads. Head 0 on, head 1 off.
            machine.fma_scalar(0, 4, 0.5, 1, 0b01).await;

            let out = vram.read(0).await;
            *got_task.lock().unwrap() = Some(tensor_values(out.as_tensor()));
        });
        executor.enter(Instant::ETERNITY).await;

        let dst = got.lock().unwrap().take().unwrap();
        assert_eq!(dst, vec![6.0, 12.0, 3.0, 4.0]);
    }

    #[tokio::test]
    async fn test_softplus_matches_reference_including_the_large_magnitude_tails() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(4, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 4, 2);
            let ty = MxDataType::Plain(fp_type);

            // -100 and +100 both overflow a naive log1p(exp(x)) in bf16: exp(100) is
            // inf, and exp(-100) underflows. The relu + log1p(exp(-|x|)) form must
            // return ~0 and ~100 respectively rather than NaN/inf.
            let input = Tensor::from_slice(&[-100.0f32, -1.0, 0.0, 100.0]);
            vram.write(0, QuantTensor::quantize(input, ty)).await;

            machine.softplus(4, 0, 0, 0).await;

            let out = vram.read(4).await;
            *got_task.lock().unwrap() = Some(tensor_values(out.as_tensor()));
        });
        executor.enter(Instant::ETERNITY).await;

        let out = got.lock().unwrap().take().unwrap();
        // softplus(-100) = 3.7e-44 -> 0 in bf16; softplus(0) = ln 2 = 0.6931;
        // softplus(-1) = 0.3133; softplus(100) = 100 to well beyond bf16 precision.
        assert!(
            out[0].abs() < 1e-30,
            "softplus(-100) should flush to ~0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.3133).abs() < 0.01,
            "softplus(-1), got {}",
            out[1]
        );
        assert!(
            (out[2] - std::f32::consts::LN_2).abs() < 0.01,
            "softplus(0) is ln 2, got {}",
            out[2]
        );
        assert!(
            (out[3] - 100.0).abs() < 1.0,
            "softplus(100) should pass through, got {}",
            out[3]
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "softplus must never produce inf/NaN: {out:?}"
        );
    }

    #[tokio::test]
    async fn test_softplus_honors_the_per_head_mask() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(4, 4, fp_type, 4));
            // tile_size 4, mask_unit 2 -> two heads of two lanes each.
            let machine = VectorMachine::new(vram.clone(), 4, 2);
            let ty = MxDataType::Plain(fp_type);

            let input = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0]);
            vram.write(0, QuantTensor::quantize(input.copy(), ty)).await;
            vram.write(4, QuantTensor::quantize(input, ty)).await;

            // mask = 0b01 -> only head 0 (lanes 0..2) is updated.
            machine.softplus(4, 0, 1, 0b01).await;

            let out = vram.read(4).await;
            *got_task.lock().unwrap() = Some(tensor_values(out.as_tensor()));
        });
        executor.enter(Instant::ETERNITY).await;

        let out = got.lock().unwrap().take().unwrap();
        assert!(
            (out[0] - std::f32::consts::LN_2).abs() < 0.01,
            "head 0 lane 0 must be softplus(0) = ln 2"
        );
        assert!(
            (out[1] - std::f32::consts::LN_2).abs() < 0.01,
            "head 0 lane 1 must be softplus(0) = ln 2"
        );
        assert_eq!(out[2], 0.0, "head 1 must be left untouched by the mask");
        assert_eq!(out[3], 0.0, "head 1 must be left untouched by the mask");
    }

    #[tokio::test]
    async fn test_vector_read_fp_round_trips_with_vector_transfer_fp() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(4, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 4, 2);

            // FP_MEM -> VRAM (S_MAP_V_FP) then VRAM -> FP_MEM (S_MAP_FP_V) must be
            // the identity: this is the contract the Mamba decay-scalar path relies on.
            let source: Vec<bf16> = [1.5f32, -2.25, 0.0, 7.0]
                .iter()
                .map(|v| bf16::from_f32(*v))
                .collect();
            machine.vector_transfer_fp(0, &source).await;
            let back = machine.vector_read_fp(0).await;

            *got_task.lock().unwrap() = Some((source, back));
        });
        executor.enter(Instant::ETERNITY).await;

        let (source, back) = got.lock().unwrap().take().unwrap();
        assert_eq!(
            source, back,
            "S_MAP_FP_V must invert S_MAP_V_FP exactly for bf16 values"
        );
    }

    #[tokio::test]
    async fn test_vector_scalar_minmax_clamps_bf16_boundary_values() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(4, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 4, 2);
            let ty = MxDataType::Plain(fp_type);

            let input = Tensor::from_slice(&[-7.03125f32, -7.0, 7.0, 7.03125]);
            vram.write(0, QuantTensor::quantize(input, ty)).await;

            machine.max_scalar(4, 0, -7.0, 0, 0).await;
            machine.min_scalar(8, 0, 7.0, 0, 0).await;

            let max_out = vram.read(4).await;
            let min_out = vram.read(8).await;
            *got_task.lock().unwrap() = Some((
                tensor_values(max_out.as_tensor()),
                tensor_values(min_out.as_tensor()),
            ));
        });

        executor.enter(Instant::ETERNITY).await;
        let (max_out, min_out) = got.lock().unwrap().take().unwrap();

        assert_eq!(max_out, vec![-7.0, -7.0, 7.0, 7.03125]);
        assert_eq!(min_out, vec![-7.03125, -7.0, 7.0, 7.0]);
    }

    #[tokio::test]
    async fn test_topk_softmax_uses_descending_logits_and_low_index_ties() {
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
            input[31] = 0.5;
            vram.write(0, QuantTensor::quantize(Tensor::from_slice(&input), ty))
                .await;

            let (indices, weights) = machine.topk_softmax(0, 32, 4).await;
            *got_task.lock().unwrap() = Some((
                indices,
                weights.into_iter().map(f32::from).collect::<Vec<_>>(),
            ));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();

        assert_eq!(indices, vec![3, 7, 9, 0]);
        let denom = 1.0 + 1.0 + f32::exp(-2.0) + f32::exp(-3.0);
        let expected = vec![
            1.0 / denom,
            1.0 / denom,
            f32::exp(-2.0) / denom,
            f32::exp(-3.0) / denom,
        ];
        for (got, exp) in weights.iter().zip(expected) {
            assert!((got - exp).abs() < 0.003, "got={got} expected={exp}");
        }
    }

    #[tokio::test]
    async fn test_topk_softmax_all_nan_row_yields_zero_weights_not_nan() {
        // A whole router-logit row of NaN maps every logit to NEG_INFINITY, so
        // max_logit is -inf and the softmax denominator is NaN. Weights must come
        // out 0.0, not NaN (regression for the finite-denominator guard).
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

            let (indices, weights) = machine.topk_softmax(0, 32, 4).await;
            *got_task.lock().unwrap() = Some((
                indices,
                weights.into_iter().map(f32::from).collect::<Vec<_>>(),
            ));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();

        assert_eq!(indices.len(), 4);
        for w in weights {
            assert!(w == 0.0, "expected 0.0 weight for an all-NaN row, got {w}");
        }
    }

    #[tokio::test]
    async fn test_topk_softmax_scans_contiguous_vector_rows_for_qwen128() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 64, 16);
            let ty = MxDataType::Plain(fp_type);

            let mut row0 = vec![-100.0f32; 64];
            let mut row1 = vec![-100.0f32; 64];
            row0[63] = 3.0;
            row1[0] = 4.0; // expert 64
            row1[7] = 5.0; // expert 71
            row1[63] = 6.0; // expert 127
            row0[2] = 7.0;
            row0[5] = 7.0; // low-index tie should pick 2 before 5
            row1[20] = 2.5; // expert 84
            row1[21] = 2.25; // expert 85
            // non-selected mass must not affect normalized selected weights.
            row0[10..30].fill(2.0);

            vram.write(0, QuantTensor::quantize(Tensor::from_slice(&row0), ty))
                .await;
            vram.write(64, QuantTensor::quantize(Tensor::from_slice(&row1), ty))
                .await;

            let (indices, weights) = machine.topk_softmax(0, 128, 8).await;
            *got_task.lock().unwrap() = Some((
                indices,
                weights.into_iter().map(f32::from).collect::<Vec<_>>(),
            ));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();

        assert_eq!(indices, vec![2, 5, 127, 71, 64, 63, 84, 85]);
        let max_logit = 7.0f32;
        let selected_logits = [7.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.25];
        let selected_exp_values: Vec<f32> = selected_logits
            .iter()
            .map(|value| (*value - max_logit).exp())
            .collect();
        let denom: f32 = selected_exp_values.iter().sum();
        for (got, exp) in weights
            .iter()
            .zip(selected_exp_values.iter().map(|value| value / denom))
        {
            assert!((got - exp).abs() < 0.003, "got={got} expected={exp}");
        }
    }
}
