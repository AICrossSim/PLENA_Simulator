use anyhow::Result;
use tch::Tensor;

use crate::dtype::{DataType, FpType, MxDataType};

fn round_ties_to_even_nonnegative(x: f32) -> u32 {
    debug_assert!(x >= 0.0);
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        (floor as u32) + 1
    } else if frac < 0.5 {
        floor as u32
    } else {
        let floor_int = floor as u32;
        if floor_int & 1 == 0 {
            floor_int
        } else {
            floor_int + 1
        }
    }
}

fn pack_codes(codes: &[u32], width: u8) -> Vec<u8> {
    assert!(width > 0 && width <= 32);
    let mut out = vec![0u8; (codes.len() * width as usize).div_ceil(8)];
    let mask = if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    let mut bit_offset = 0usize;
    for code in codes {
        let value = code & mask;
        for bit in 0..width as usize {
            if value & (1u32 << bit) != 0 {
                let position = bit_offset + bit;
                out[position / 8] |= 1u8 << (position % 8);
            }
        }
        bit_offset += width as usize;
    }
    out
}

fn mxint_bits_from_scaled(value: f32, width: u8) -> u32 {
    assert!(width >= 2 && width <= 32);
    let magnitude_bits = width - 1;
    let magnitude_max = if magnitude_bits == 31 {
        i32::MAX as u32
    } else {
        (1u32 << magnitude_bits) - 1
    };
    let magnitude = round_ties_to_even_nonnegative(value.abs() * (1u64 << magnitude_bits) as f32)
        .min(magnitude_max);
    let sign = u32::from(value.is_sign_negative() && magnitude != 0);
    (sign << magnitude_bits) | magnitude
}

fn mxint_scaled_from_bits(bits: u32, width: u8) -> f32 {
    assert!(width >= 2 && width <= 32);
    let magnitude_bits = width - 1;
    let magnitude_mask = if magnitude_bits == 31 {
        i32::MAX as u32
    } else {
        (1u32 << magnitude_bits) - 1
    };
    let magnitude = bits & magnitude_mask;
    if magnitude == 0 {
        return 0.0;
    }
    let sign = if (bits >> magnitude_bits) & 1 == 1 {
        -1.0
    } else {
        1.0
    };
    sign * magnitude as f32 / (1u64 << magnitude_bits) as f32
}

/// Quantize one FP32 value with the shared saturating RNE minifloat semantics.
fn minifloat_ieee_quantize_hardware(value: f32, fp_type: FpType) -> u32 {
    fp_type.bits_from_f32(value)
}

pub struct QuantTensor {
    tensor: Tensor,
    ty: MxDataType,
}

fn validate_quant_type(ty: MxDataType) -> Result<()> {
    if let MxDataType::Mx {
        elem: DataType::Int(int_ty),
        ..
    } = ty
    {
        anyhow::ensure!(
            matches!(int_ty.width, 2 | 4 | 8),
            "MXINT element width must be 2, 4, or 8 bits"
        );
    }
    Ok(())
}

impl Clone for QuantTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.copy(),
            ty: self.ty,
        }
    }
}

impl QuantTensor {
    /// Create a quantized tensor, assuming the tensor is already quantized.
    pub fn new_assuming_quantized(tensor: Tensor, ty: MxDataType) -> Result<Self> {
        anyhow::ensure!(tensor.dim() == 1);
        anyhow::ensure!(tensor.kind() == tch::Kind::Float);
        anyhow::ensure!(tensor.device() == tch::Device::Cpu);
        validate_quant_type(ty)?;
        Ok(QuantTensor { tensor: tensor, ty })
    }

    /// Create a quantized tensor, assuming the tensor is already quantized.
    pub fn quantize(tensor: Tensor, ty: MxDataType) -> Self {
        // Physical quantization occurs when the tensor is serialized.
        Self::new_assuming_quantized(tensor, ty).unwrap()
    }

    /// Quantize and immediately decode a one-dimensional tensor.
    pub fn quantize_materialized(tensor: Tensor, ty: MxDataType) -> Self {
        let len = tensor.size1().unwrap() as usize;
        let mut encoded = Self::new_assuming_quantized(tensor, ty).unwrap();
        let (element_bytes, scale_bytes) = encoded.into_bytes();
        Self::from_bytes(&element_bytes, &scale_bytes, len, ty)
    }

    /// Create a zeroed quantized tensor.
    pub fn zeros(size: usize, ty: MxDataType) -> Self {
        Self::new_assuming_quantized(
            Tensor::zeros([size as i64], (tch::Kind::Float, tch::Device::Cpu)),
            ty,
        )
        .unwrap()
    }

    /// Return the underlying torch Tensor.
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Return the data type of the quantized tensor.
    pub fn data_type(&self) -> MxDataType {
        self.ty
    }

    /// Deserialize a quantized tensor from bytes.
    pub fn from_bytes(bytes: &[u8], scale_bytes: &[u8], len: usize, ty: MxDataType) -> Self {
        validate_quant_type(ty).expect("unsupported quantized tensor type");
        let elem_ty = ty.element_type();
        let element_byte_count = (len * elem_ty.size_in_bits() as usize).div_ceil(8);
        assert_eq!(
            bytes.len(),
            element_byte_count,
            "element byte plane length does not match the tensor"
        );

        let mut vec = vec![0f32; len];
        elem_ty.convert_bytes_to_f32_vec(bytes, &mut vec);
        if let DataType::Int(int_ty) = elem_ty {
            for value in &mut vec {
                *value = mxint_scaled_from_bits(*value as u32, int_ty.size_in_bits());
            }
        }

        if let MxDataType::Mx {
            elem: _,
            scale,
            block,
        } = ty
        {
            assert!(len.is_multiple_of(block as usize));
            let scale_count = len / block as usize;
            let scale_byte_count = (scale_count * scale.size_in_bits() as usize).div_ceil(8);
            assert_eq!(
                scale_bytes.len(),
                scale_byte_count,
                "scale byte plane length does not match the tensor"
            );
            if scale == DataType::Fp(FpType::E8M0) {
                assert_eq!(scale.size_in_bits(), 8);
                for (elements, code) in vec
                    .chunks_mut(block as usize)
                    .zip(scale_bytes.iter().copied())
                {
                    let exponent = code as i32 - 127;
                    let multiplier = 2.0f64.powi(exponent);
                    for element in elements {
                        *element = (*element as f64 * multiplier) as f32;
                    }
                }
            } else {
                let mut scale_vec = vec![0f32; scale_count];
                scale.convert_bytes_to_f32_vec(scale_bytes, &mut scale_vec);
                for (elements, scale_value) in vec
                    .chunks_mut(block as usize)
                    .zip(scale_vec.iter().copied())
                {
                    for element in elements {
                        *element *= scale_value;
                    }
                }
            }
        } else {
            assert!(
                scale_bytes.is_empty(),
                "plain tensors cannot carry a scale plane"
            );
        }

        let tensor = tch::Tensor::from_slice(&vec);
        Self { tensor, ty }
    }

    /// Serialize the quantized tensor into bytes.
    pub fn into_bytes(&mut self) -> (Vec<u8>, Vec<u8>) {
        let len = self.tensor.size1().unwrap() as usize;
        let slice =
            unsafe { core::slice::from_raw_parts(self.tensor.data_ptr() as *const f32, len) };
        assert!(
            slice.iter().all(|value| value.is_finite()),
            "cannot serialize a non-finite quantized tensor"
        );
        tracing::trace!("slice: {:?}", slice);

        let elem_ty = self.ty.element_type();

        if let MxDataType::Mx { elem, scale, block } = self.ty {
            // Properly calculate MX scales and quantize elements
            assert!(len.is_multiple_of(block as usize));
            let num_blocks = len / block as usize;
            let mut element_codes = Vec::with_capacity(len);
            let mut scale_codes = Vec::with_capacity(num_blocks);

            // Process each block
            for (block_idx, block_data) in slice.chunks(block as usize).enumerate() {
                tracing::trace!("block_idx: {}", block_idx);
                tracing::trace!("block_data: {:?}", block_data);
                // Find maximum absolute value in this block
                let max_abs = block_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                let scale_exp_bits = match scale {
                    DataType::Fp(scale_fp) => scale_fp.exponent,
                    _ => 8,
                };
                let scale_bias = (1u32 << (scale_exp_bits - 1)) - 1;

                if max_abs == 0.0 {
                    scale_codes.push(scale_bias);
                    element_codes.extend(core::iter::repeat(0).take(block as usize));
                } else {
                    let scale_exp_min = -(scale_bias as i32);
                    let scale_exp_max = ((1u32 << scale_exp_bits) - 1 - scale_bias) as i32;
                    let raw_exp = match elem {
                        DataType::Int(int_ty) => {
                            let magnitude_bits = int_ty.size_in_bits() - 1;
                            let qmax = ((1u64 << magnitude_bits) - 1) as f32
                                / (1u64 << magnitude_bits) as f32;
                            (max_abs / qmax).log2().ceil() as i32
                        }
                        DataType::Fp(elem_fp) => {
                            // OCP MX places the block maximum in the element
                            // format's highest finite exponent bin. The E8M0
                            // scale is explicit data consumed by the datapath.
                            let element_max_exponent = elem_fp.max_finite_exponent_code() as i32
                                - elem_fp.exponent_bias() as i32;
                            max_abs.log2().floor() as i32 - element_max_exponent
                        }
                    };
                    let per_block_exponent_bias = raw_exp.max(scale_exp_min).min(scale_exp_max);
                    let stored_scale = per_block_exponent_bias + scale_bias as i32;
                    scale_codes.push(stored_scale as u32);

                    for value in block_data {
                        let scaled = *value * 2.0f32.powi(-per_block_exponent_bias);
                        let code = match elem {
                            DataType::Fp(elem_fp) => {
                                minifloat_ieee_quantize_hardware(scaled, elem_fp)
                            }
                            DataType::Int(int_ty) => {
                                mxint_bits_from_scaled(scaled, int_ty.size_in_bits())
                            }
                        };
                        element_codes.push(code);
                    }
                }
            }

            let out = pack_codes(&element_codes, elem.size_in_bits());
            let scale_out = pack_codes(&scale_codes, scale.size_in_bits());
            tracing::trace!("scale_out: {:?}", scale_out);
            return (out, scale_out);
        }

        // Plain type: no scales
        let mut out = vec![0; (len * elem_ty.size_in_bits() as usize).div_ceil(8)];
        elem_ty.bytes_from_f32(slice, &mut out);
        (out, Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{DataType, FpType, IntType, MxDataType};
    use tch::Tensor;

    fn e4m3() -> FpType {
        FpType {
            sign: true,
            exponent: 4,
            mantissa: 3,
        }
    }

    #[test]
    fn test_minifloat_quantize_known_values() {
        let ty = e4m3();
        // Exactly-representable values (no rounding-boundary ambiguity).
        assert_eq!(minifloat_ieee_quantize_hardware(0.0, ty), 0);
        assert_eq!(minifloat_ieee_quantize_hardware(1.0, ty), 0x38); // exp 0, mantissa 0
        assert_eq!(minifloat_ieee_quantize_hardware(1.875, ty), 0x3F); // exp 0, mantissa 7
        assert_eq!(minifloat_ieee_quantize_hardware(-1.0, ty), 0xB8); // sign | 0x38
    }

    #[test]
    fn test_minifloat_quantize_hardware_rounds_fractional_mantissa() {
        let ty = e4m3();
        // The conversion keeps guard and sticky information and rounds to even.
        assert_eq!(minifloat_ieee_quantize_hardware(1.0625, ty), 0x38);
        assert_eq!(minifloat_ieee_quantize_hardware(1.0859375, ty), 0x39);
        assert_eq!(minifloat_ieee_quantize_hardware(1.1875, ty), 0x3A);
        assert_eq!(minifloat_ieee_quantize_hardware(1.328125, ty), 0x3B);
        assert_eq!(minifloat_ieee_quantize_hardware(-1.0625, ty), 0xB8);
        assert_eq!(minifloat_ieee_quantize_hardware(-1.1875, ty), 0xBA);
        assert_eq!(minifloat_ieee_quantize_hardware(1.9375, ty), 0x40);
        assert_eq!(minifloat_ieee_quantize_hardware(0.0146484375, ty), 0x08);
        assert_eq!(minifloat_ieee_quantize_hardware(256.0, ty), 0x77);
    }

    #[test]
    fn test_into_bytes_plain_packs_elements_and_no_scale_stream() {
        let elem = DataType::Fp(e4m3());
        let t = Tensor::from_slice(&[0.0f32, 0.5, 1.0, 1.875, -1.0]);
        let mut qt = QuantTensor::new_assuming_quantized(t, MxDataType::Plain(elem)).unwrap();
        let (bytes, scale_bytes) = qt.into_bytes();
        assert_eq!(bytes, vec![0, 48, 56, 63, 184]);
        assert!(scale_bytes.is_empty()); // plain type emits no scale stream
    }

    #[test]
    fn test_into_bytes_mx_block_scale() {
        // E4M3 has maximum unbiased element exponent 7. OCP placement maps a
        // block maximum of 4.0 to shared exponent 2 - 7 = -5 (stored as 122).
        let ty = MxDataType::Mx {
            elem: DataType::Fp(e4m3()),
            scale: DataType::Fp(FpType::E8M0),
            block: 4,
        };
        let t = Tensor::from_slice(&[0.5f32, 1.0, 2.0, 4.0]);
        let mut qt = QuantTensor::new_assuming_quantized(t, ty).unwrap();
        let (bytes, scale_bytes) = qt.into_bytes();
        assert_eq!(bytes, vec![88, 96, 104, 112]);
        assert_eq!(scale_bytes, vec![122]);
    }

    #[test]
    fn test_mxfp_sweep_formats_use_ocp_scales_and_are_idempotent() {
        let formats = [
            (
                FpType {
                    sign: true,
                    exponent: 1,
                    mantissa: 2,
                },
                129u8,
            ),
            (
                FpType {
                    sign: true,
                    exponent: 2,
                    mantissa: 1,
                },
                128u8,
            ),
            (
                FpType {
                    sign: true,
                    exponent: 3,
                    mantissa: 4,
                },
                126u8,
            ),
            (
                FpType {
                    sign: true,
                    exponent: 4,
                    mantissa: 3,
                },
                122u8,
            ),
            (
                FpType {
                    sign: true,
                    exponent: 5,
                    mantissa: 2,
                },
                114u8,
            ),
        ];
        for (elem, expected_scale) in formats {
            let ty = MxDataType::Mx {
                elem: DataType::Fp(elem),
                scale: DataType::Fp(FpType::E8M0),
                block: 8,
            };
            let tensor = Tensor::from_slice(&[0.5f32, 1.0, 2.0, 4.0, 0.0, -0.0, 0.25, -0.25]);
            let mut first = QuantTensor::new_assuming_quantized(tensor, ty).unwrap();
            let (first_elements, first_scales) = first.into_bytes();
            assert_eq!(first_scales, vec![expected_scale], "element {elem:?}");

            let mut decoded = QuantTensor::from_bytes(&first_elements, &first_scales, 8, ty);
            let (second_elements, second_scales) = decoded.into_bytes();
            assert_eq!(first_elements, second_elements, "element {elem:?}");
            assert_eq!(first_scales, second_scales, "element {elem:?}");
        }
    }

    #[test]
    fn test_into_bytes_zero_block_uses_unit_scale() {
        let ty = MxDataType::Mx {
            elem: DataType::Fp(e4m3()),
            scale: DataType::Fp(FpType::E8M0),
            block: 4,
        };
        let tensor = Tensor::from_slice(&[0.0f32; 4]);
        let mut quantized = QuantTensor::new_assuming_quantized(tensor, ty).unwrap();
        let (elements, scales) = quantized.into_bytes();
        assert_eq!(elements, vec![0; 4]);
        assert_eq!(scales, vec![127]);
    }

    #[test]
    fn test_mxint_signed_subbyte_roundtrip() {
        for width in [2u32, 4, 8] {
            let ty = MxDataType::Mx {
                elem: DataType::Int(IntType { width }),
                scale: DataType::Fp(FpType::E8M0),
                block: 8,
            };
            let input = [-0.75f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.875];
            let tensor = Tensor::from_slice(&input);
            let mut quantized = QuantTensor::new_assuming_quantized(tensor, ty).unwrap();
            let (elements, scales) = quantized.into_bytes();
            assert_eq!(elements.len(), width as usize);
            assert_eq!(scales.len(), 1);
            let decoded = QuantTensor::from_bytes(&elements, &scales, 8, ty);
            let values: Vec<f32> = Vec::<f32>::try_from(decoded.as_tensor()).unwrap();
            for (actual, expected) in values.iter().zip(input) {
                let tolerance = 1.0 / (1u32 << (width - 1)) as f32;
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn test_mxint_sign_magnitude_extrema_and_canonical_zero() {
        for width in [2u8, 4, 8] {
            let magnitude_bits = width - 1;
            let magnitude_max = (1u32 << magnitude_bits) - 1;
            let unit = 1.0 / (1u32 << magnitude_bits) as f32;
            assert_eq!(mxint_bits_from_scaled(0.0, width), 0);
            assert_eq!(mxint_bits_from_scaled(-0.0, width), 0);
            assert_eq!(mxint_bits_from_scaled(unit, width), 1);
            assert_eq!(
                mxint_bits_from_scaled(-unit, width),
                (1u32 << magnitude_bits) | 1
            );
            assert_eq!(mxint_bits_from_scaled(1.0, width), magnitude_max);
            assert_eq!(
                mxint_bits_from_scaled(-1.0, width),
                (1u32 << magnitude_bits) | magnitude_max
            );
            assert_eq!(
                mxint_scaled_from_bits(magnitude_max, width),
                magnitude_max as f32 * unit
            );
            assert_eq!(
                mxint_scaled_from_bits((1u32 << magnitude_bits) | magnitude_max, width,),
                -(magnitude_max as f32) * unit
            );
            let negative_zero = mxint_scaled_from_bits(1u32 << magnitude_bits, width);
            assert_eq!(negative_zero.to_bits(), 0.0f32.to_bits());
        }
    }

    #[test]
    fn test_mxint_physical_codes_at_unit_scale() {
        for (width, expected) in [
            (2u32, vec![0xdd, 0x00]),
            (4u32, vec![0x91, 0xf7, 0x00, 0x00]),
            (8u32, vec![1, 129, 127, 255, 0, 0, 0, 0]),
        ] {
            let ty = MxDataType::Mx {
                elem: DataType::Int(IntType { width }),
                scale: DataType::Fp(FpType::E8M0),
                block: 8,
            };
            let magnitude_bits = width - 1;
            let unit = 1.0 / (1u32 << magnitude_bits) as f32;
            let qmax = ((1u32 << magnitude_bits) - 1) as f32 * unit;
            let tensor = Tensor::from_slice(&[unit, -unit, qmax, -qmax, 0.0, -0.0, 0.0, 0.0]);
            let mut quantized = QuantTensor::new_assuming_quantized(tensor, ty).unwrap();
            let (elements, scales) = quantized.into_bytes();
            assert_eq!(elements, expected, "width {width}");
            assert_eq!(scales, vec![127], "width {width}");
        }
    }

    #[test]
    fn test_e8m0_maximum_exponent_decodes_without_intermediate_infinity() {
        let ty = MxDataType::Mx {
            elem: DataType::Int(IntType { width: 2 }),
            scale: DataType::Fp(FpType::E8M0),
            block: 8,
        };
        let decoded = QuantTensor::from_bytes(&[0x01, 0x00], &[255], 8, ty);
        let values: Vec<f32> = Vec::<f32>::try_from(decoded.as_tensor()).unwrap();
        assert_eq!(values[0], 2.0f32.powi(127));
        assert!(values[1..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_mxint_scale_is_range_safe_and_idempotent() {
        for width in [2u32, 4, 8] {
            let ty = MxDataType::Mx {
                elem: DataType::Int(IntType { width }),
                scale: DataType::Fp(FpType::E8M0),
                block: 8,
            };
            let input = Tensor::from_slice(&[-1.0f32, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]);
            let mut first = QuantTensor::new_assuming_quantized(input, ty).unwrap();
            let (first_elements, first_scales) = first.into_bytes();
            let mut decoded = QuantTensor::from_bytes(&first_elements, &first_scales, 8, ty);
            let (second_elements, second_scales) = decoded.into_bytes();
            assert_eq!(first_elements, second_elements, "width {width}");
            assert_eq!(first_scales, second_scales, "width {width}");
            let values = QuantTensor::from_bytes(&first_elements, &first_scales, 8, ty);
            let values: Vec<f32> = Vec::<f32>::try_from(values.as_tensor()).unwrap();
            assert!(values.iter().all(|value| value.abs() <= 1.0));
            assert!(values.iter().any(|value| *value == 1.0));
            assert!(values.iter().any(|value| *value == -1.0));
        }
    }

    #[test]
    fn test_quantize_materialized_applies_mxint_rounding() {
        let ty = MxDataType::Mx {
            elem: DataType::Int(IntType { width: 2 }),
            scale: DataType::Fp(FpType::E8M0),
            block: 8,
        };
        let input = Tensor::from_slice(&[-0.40f32, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40]);
        let rounded = QuantTensor::quantize_materialized(input, ty);
        let values: Vec<f32> = Vec::<f32>::try_from(rounded.as_tensor()).unwrap();
        assert_eq!(values, vec![-0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn test_mxint_rejects_non_power_of_two_width() {
        let ty = MxDataType::Mx {
            elem: DataType::Int(IntType { width: 3 }),
            scale: DataType::Fp(FpType::E8M0),
            block: 8,
        };
        let tensor = Tensor::zeros([8], (tch::Kind::Float, tch::Device::Cpu));
        assert!(QuantTensor::new_assuming_quantized(tensor, ty).is_err());
    }

    #[test]
    fn test_pack_codes_little_endian_across_bytes() {
        assert_eq!(pack_codes(&[1, 2, 3], 4), vec![0x21, 0x03]);
    }

    #[test]
    fn test_from_bytes_plain_then_into_bytes_roundtrip() {
        // Decode three e4m3 bytes, then re-serialize to a stable byte stream.
        let ty = MxDataType::Plain(DataType::Fp(e4m3()));
        let mut qt = QuantTensor::from_bytes(&[0x38u8, 0x3F, 0x00], &[], 3, ty);
        let (out_bytes, _) = qt.into_bytes();
        assert_eq!(out_bytes, vec![0x38, 0x3F, 0x00]);
    }
}
