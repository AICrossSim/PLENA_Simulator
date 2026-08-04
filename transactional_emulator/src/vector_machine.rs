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
use quantize::QuantTensor;
use sram::VectorSram;
use tch::Tensor;

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
}

#[cfg(test)]
mod tests {
    use super::head_is_selected;

    #[test]
    fn head_mask_selects_only_requested_segments() {
        let mask = 0b0101;
        assert!(head_is_selected(mask, 0));
        assert!(!head_is_selected(mask, 1));
        assert!(head_is_selected(mask, 2));
        assert!(!head_is_selected(mask, 3));
        assert!(!head_is_selected(u32::MAX, 32));
    }
}
