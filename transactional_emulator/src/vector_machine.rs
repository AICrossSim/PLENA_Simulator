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
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let val: f32 = result.max().try_into().unwrap();
            f32::max(val, f)
        }
    }

    fn segment_bounds(&self, segment_index: u32, segment_log2: u8) -> (i64, i64) {
        let width = 1_u32
            .checked_shl(u32::from(segment_log2))
            .expect("segment reduction width overflows u32");
        let start = segment_index
            .checked_mul(width)
            .expect("segment reduction offset overflows u32");
        let end = start
            .checked_add(width)
            .expect("segment reduction end overflows u32");
        assert!(
            end <= self.tile_size,
            "segment reduction [{start}, {end}) exceeds VLEN {}",
            self.tile_size
        );
        (i64::from(start), i64::from(width))
    }

    pub(crate) async fn reduce_sum_segment(
        &self,
        vs1: u32,
        seed: f32,
        segment_index: u32,
        segment_log2: u8,
    ) -> f32 {
        let a = self.vram.read(vs1).await;
        let (start, width) = self.segment_bounds(segment_index, segment_log2);
        cycle!(*VECTOR_SUM_CYCLES);
        let value: f32 = a
            .as_tensor()
            .narrow(0, start, width)
            .sum(tch::Kind::Float)
            .try_into()
            .unwrap();
        seed + value
    }

    pub(crate) async fn reduce_max_segment(
        &self,
        vs1: u32,
        seed: f32,
        segment_index: u32,
        segment_log2: u8,
    ) -> f32 {
        let a = self.vram.read(vs1).await;
        let (start, width) = self.segment_bounds(segment_index, segment_log2);
        cycle!(*VECTOR_MAX_CYCLES);
        let value: f32 = a
            .as_tensor()
            .narrow(0, start, width)
            .max()
            .try_into()
            .unwrap();
        f32::max(seed, value)
    }

    async fn reduce_segments(&self, vd: u32, vs1: u32, segment_log2: u8, is_max: bool) {
        let source = self.vram.read(vs1).await;
        let width = 1_u32
            .checked_shl(u32::from(segment_log2))
            .expect("multi-segment reduction width overflows u32");
        assert!(width > 0 && self.tile_size % width == 0);
        let count = self.tile_size / width;
        assert!(
            count <= 64,
            "RTL-v5 supports at most 64 segments, got {count}"
        );

        let mut level = source
            .as_tensor()
            .reshape([i64::from(count), i64::from(width)]);
        let mut level_width = i64::from(width);
        while level_width > 1 {
            let pairs = level.reshape([i64::from(count), level_width / 2, 2]);
            let next = if is_max {
                pairs.max_dim(2, false).0
            } else {
                pairs.sum_dim_intlist([2_i64].as_slice(), false, tch::Kind::Float)
            };
            // Match the registered RTL tree: every level rounds back to the
            // configured VectorMachine FP representation before the next level.
            let next_shape = next.size();
            let quantized = QuantTensor::quantize(next.reshape([-1]), source.data_type());
            level = quantized.as_tensor().reshape(next_shape.as_slice());
            level_width /= 2;
        }

        let compact = Tensor::zeros(
            [i64::from(self.tile_size)],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        compact
            .narrow(0, 0, i64::from(count))
            .copy_(&level.reshape([i64::from(count)]));
        let result = QuantTensor::quantize(compact, source.data_type());
        cycle!(*VECTOR_SUM_CYCLES);
        self.vram.write(vd, result).await;
    }

    pub(crate) async fn reduce_sum_segments(&self, vd: u32, vs1: u32, segment_log2: u8) {
        self.reduce_segments(vd, vs1, segment_log2, false).await;
    }

    pub(crate) async fn reduce_max_segments(&self, vd: u32, vs1: u32, segment_log2: u8) {
        self.reduce_segments(vd, vs1, segment_log2, true).await;
    }

    pub(crate) async fn alu_vseg(
        &self,
        vd: u32,
        vs1: u32,
        vstats: u32,
        segment_log2: u8,
        operation: u8,
        mask_enable: bool,
        mask: u32,
    ) {
        let (source, stats) = tokio::join!(self.vram.read(vs1), self.vram.read(vstats));
        let width = 1_i64 << segment_log2;
        let count = i64::from(self.tile_size) / width;
        assert!(
            count <= 64 && count * width == i64::from(self.tile_size),
            "VSEG supports at most 64 segments, got {count}"
        );
        let broadcast = stats
            .as_tensor()
            .narrow(0, 0, count)
            .unsqueeze(1)
            .repeat([1, width])
            .reshape([i64::from(self.tile_size)]);
        let computed = match operation {
            0 => source.as_tensor() + &broadcast,
            1 => source.as_tensor() - &broadcast,
            2 => source.as_tensor() * &broadcast,
            other => panic!("invalid V_ALU_VSEG operation {other}"),
        };
        let result = if mask_enable {
            assert!(
                count <= 32,
                "masked VSEG is limited by the 32-bit architectural mask; \
                 64-segment full blocks must use the unmasked form"
            );
            let merged = source.as_tensor().shallow_clone();
            for segment in 0..count {
                if mask & (1_u32 << segment) != 0 {
                    merged
                        .narrow(0, segment * width, width)
                        .copy_(&computed.narrow(0, segment * width, width));
                }
            }
            merged
        } else {
            computed
        };
        let quantized = QuantTensor::quantize(result, source.data_type());
        cycle!(if operation == 2 {
            *VECTOR_MUL_CYCLES
        } else {
            *VECTOR_ADD_CYCLES
        });
        self.vram.write(vd, quantized).await;
    }

    pub(crate) async fn compact_stats(
        &self,
        vd: u32,
        vs1: u32,
        scalar: f32,
        segment_count: u8,
        operation: u8,
    ) {
        let source = self.vram.read(vs1).await;
        let count = i64::from(segment_count);
        assert!(
            (1..=64).contains(&segment_count) && count <= i64::from(self.tile_size),
            "compact-stat lane count must be in [1, min(64, VLEN)], got {segment_count}"
        );
        let active = source.as_tensor().narrow(0, 0, count);
        let computed = match operation {
            0 => QuantTensor::quantize(active * (scalar as f64), source.data_type()),
            1 => QuantTensor::quantize(active + (scalar as f64), source.data_type()),
            2 => {
                // Match S_RSQRT_FP: round sqrt before the reciprocal stage.
                let sqrt = QuantTensor::quantize(active.sqrt(), source.data_type());
                QuantTensor::quantize(sqrt.as_tensor().reciprocal(), source.data_type())
            }
            other => panic!("invalid compact-stat operation {other}"),
        };
        let result = Tensor::zeros(
            [i64::from(self.tile_size)],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        result
            .narrow(0, 0, count)
            .copy_(&computed.as_tensor().narrow(0, 0, count));
        cycle!(if operation == 0 {
            *VECTOR_MUL_CYCLES
        } else {
            *VECTOR_ADD_CYCLES
        });
        self.vram
            .write(vd, QuantTensor::quantize(result, source.data_type()))
            .await;
    }

    pub(crate) async fn load_lane(&self, vs1: u32, lane: u32) -> f32 {
        assert!(lane < self.tile_size);
        let source = self.vram.read(vs1).await;
        source
            .as_tensor()
            .double_value([i64::from(lane)].as_slice()) as f32
    }

    pub(crate) async fn store_lane(&self, vd: u32, lane: u32, value: f32) {
        assert!(lane < self.tile_size);
        let current = self.vram.read(vd).await;
        let updated = current.as_tensor().shallow_clone();
        let _ = updated.narrow(0, i64::from(lane), 1).fill_(value as f64);
        self.vram
            .write(vd, QuantTensor::quantize(updated, current.data_type()))
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{DataType, FpType};
    use runtime::{Executor, Instant};
    use std::sync::Mutex;
    use tch::{Device, Kind};

    #[tokio::test]
    async fn multi_segment_reduction_supports_64_roots() {
        let executor = Executor::new();
        let passed = Arc::new(Mutex::new(None));
        let task_result = passed.clone();
        executor.spawn(async move {
            let ty = DataType::Fp(FpType {
                sign: true,
                exponent: 8,
                mantissa: 7,
            });
            let vram = Arc::new(VectorSram::new(64, 4, ty, 64));
            let input = QuantTensor::quantize(
                Tensor::arange(64, (Kind::Float, Device::Cpu)),
                vram.ty(),
            );
            vram.write(0, input).await;
            let machine = VectorMachine::new(vram.clone(), 64, 1);

            machine.reduce_sum_segments(64, 0, 0).await;
            machine.reduce_max_segments(128, 0, 0).await;

            let source = vram.read(0).await;
            let sum = vram.read(64).await;
            let max = vram.read(128).await;
            *task_result.lock().unwrap() = Some(
                sum.as_tensor()
                    .allclose(source.as_tensor(), 0.0, 0.0, false)
                    && max
                        .as_tensor()
                        .allclose(source.as_tensor(), 0.0, 0.0, false),
            );
        });
        executor.enter(Instant::ETERNITY).await;
        assert_eq!(*passed.lock().unwrap(), Some(true));
    }
}
