use half::{bf16, f16};

use super::error::StateEngineError;
use super::generated_contract::StatePrecision;

pub const MX_BLOCK_SIZE: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedTensor {
    pub values: Vec<u8>,
    pub scales: Vec<u8>,
}

pub fn scale_count(elements: usize, inner_dim: usize) -> Result<usize, StateEngineError> {
    if inner_dim == 0 || !elements.is_multiple_of(inner_dim) {
        return Err(StateEngineError::invalid(
            "MX tensor elements must be a multiple of a nonzero inner dimension",
        ));
    }
    Ok((elements / inner_dim) * inner_dim.div_ceil(MX_BLOCK_SIZE))
}

pub fn encode_tensor(
    source: &[f32],
    precision: StatePrecision,
    inner_dim: usize,
) -> Result<EncodedTensor, StateEngineError> {
    if source.iter().any(|value| !value.is_finite()) {
        return Err(StateEngineError::internal(
            "X_STATE attempted to persist a non-finite tensor value",
        ));
    }
    let mut values = Vec::with_capacity(source.len() * precision.element_bytes());
    let mut scales = Vec::new();
    match precision {
        StatePrecision::Fp32 => {
            for value in source {
                values.extend_from_slice(&value.to_le_bytes());
            }
        }
        StatePrecision::Bf16 => {
            for value in source {
                values.extend_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
            }
        }
        StatePrecision::Fp16 => {
            for value in source {
                values.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
            }
        }
        StatePrecision::Mx8B128 => {
            let expected_scales = scale_count(source.len(), inner_dim)?;
            values.reserve(source.len());
            scales.reserve(expected_scales);
            for outer in source.chunks_exact(inner_dim) {
                for block in outer.chunks(MX_BLOCK_SIZE) {
                    let maximum = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
                    let exponent = if maximum == 0.0 {
                        0
                    } else {
                        (maximum / 448.0).log2().ceil().clamp(-126.0, 127.0) as i32
                    };
                    scales.push((exponent + 127) as u8);
                    let scale = 2.0f32.powi(exponent);
                    values.extend(block.iter().map(|value| encode_e4m3fn(*value / scale)));
                }
            }
            debug_assert_eq!(scales.len(), expected_scales);
        }
    }
    Ok(EncodedTensor { values, scales })
}

pub fn decode_tensor(
    values: &[u8],
    scales: &[u8],
    precision: StatePrecision,
    inner_dim: usize,
) -> Result<Vec<f32>, StateEngineError> {
    let element_bytes = precision.element_bytes();
    if !values.len().is_multiple_of(element_bytes) {
        return Err(StateEngineError::invalid(
            "tensor byte length is not divisible by its element size",
        ));
    }
    let elements = values.len() / element_bytes;
    if precision != StatePrecision::Mx8B128 && !scales.is_empty() {
        return Err(StateEngineError::invalid(
            "non-MX tensor unexpectedly carries scale bytes",
        ));
    }
    match precision {
        StatePrecision::Fp32 => Ok(values
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()),
        StatePrecision::Bf16 => Ok(values
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
            .collect()),
        StatePrecision::Fp16 => Ok(values
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
            .collect()),
        StatePrecision::Mx8B128 => {
            let expected_scales = scale_count(elements, inner_dim)?;
            if scales.len() != expected_scales {
                return Err(StateEngineError::invalid(format!(
                    "MX tensor has {} scale bytes, expected {expected_scales}",
                    scales.len()
                )));
            }
            let mut output = Vec::with_capacity(elements);
            let mut value_offset = 0;
            let mut scale_offset = 0;
            for _ in 0..elements / inner_dim {
                let mut remaining = inner_dim;
                while remaining > 0 {
                    let block_len = remaining.min(MX_BLOCK_SIZE);
                    let exponent = i32::from(scales[scale_offset]) - 127;
                    let scale = 2.0f32.powi(exponent);
                    output.extend(
                        values[value_offset..value_offset + block_len]
                            .iter()
                            .map(|bits| decode_e4m3fn(*bits) * scale),
                    );
                    value_offset += block_len;
                    scale_offset += 1;
                    remaining -= block_len;
                }
            }
            Ok(output)
        }
    }
}

pub fn requantize(
    source: &[f32],
    precision: StatePrecision,
    inner_dim: usize,
) -> Result<(Vec<f32>, EncodedTensor), StateEngineError> {
    let encoded = encode_tensor(source, precision, inner_dim)?;
    let restored = decode_tensor(&encoded.values, &encoded.scales, precision, inner_dim)?;
    Ok((restored, encoded))
}

fn encode_e4m3fn(value: f32) -> u8 {
    let sign = u8::from(value.is_sign_negative()) << 7;
    let magnitude = value.abs().min(448.0);
    if magnitude == 0.0 {
        return sign;
    }
    let minimum_normal = 2.0f32.powi(-6);
    if magnitude < minimum_normal {
        let quantized = (magnitude / 2.0f32.powi(-9)).round_ties_even() as u8;
        return if quantized < 8 {
            sign | quantized
        } else {
            sign | (1 << 3)
        };
    }

    let mut exponent = magnitude.log2().floor().clamp(-6.0, 8.0) as i32;
    let step = 2.0f32.powi(exponent - 3);
    let mut significand = (magnitude / step).round_ties_even() as u8;
    if significand == 16 {
        exponent += 1;
        significand = 8;
    }
    if exponent >= 8 {
        let mantissa = significand.saturating_sub(8).min(6);
        return sign | (0x0f << 3) | mantissa;
    }
    let biased_exponent = (exponent + 7) as u8;
    sign | (biased_exponent << 3) | significand.saturating_sub(8).min(7)
}

fn decode_e4m3fn(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    let magnitude = match exponent {
        0 => f32::from(mantissa) * 2.0f32.powi(-9),
        15 => 256.0 + f32::from(mantissa.min(6)) * 32.0,
        _ => (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7),
    };
    sign * magnitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip_uses_storage_precision() {
        let encoded = encode_tensor(&[1.001, -2.5], StatePrecision::Bf16, 2).unwrap();
        assert_eq!(encoded.values.len(), 4);
        assert!(encoded.scales.is_empty());
        let restored = decode_tensor(&encoded.values, &[], StatePrecision::Bf16, 2).unwrap();
        assert_eq!(restored[0], bf16::from_f32(1.001).to_f32());
        assert_eq!(restored[1], -2.5);
    }

    #[test]
    fn mx8_scales_each_inner_axis_block() {
        let mut source = vec![0.0; 2 * 129];
        source[0] = 448.0;
        source[128] = 896.0;
        source[129] = -224.0;
        let encoded = encode_tensor(&source, StatePrecision::Mx8B128, 129).unwrap();
        assert_eq!(encoded.values.len(), source.len());
        assert_eq!(encoded.scales.len(), 4);
        assert_eq!(&encoded.scales[..2], &[127, 128]);
        let restored = decode_tensor(
            &encoded.values,
            &encoded.scales,
            StatePrecision::Mx8B128,
            129,
        )
        .unwrap();
        assert_eq!(restored[0], 448.0);
        assert_eq!(restored[128], 896.0);
        assert_eq!(restored[129], -224.0);
    }

    #[test]
    fn e4m3fn_covers_subnormal_and_maximum() {
        for value in [2.0f32.powi(-9), 1.0, 240.0, 448.0, -448.0] {
            assert_eq!(decode_e4m3fn(encode_e4m3fn(value)), value);
        }
    }

    #[test]
    fn e4m3fn_rounding_carries_into_the_next_exponent() {
        assert_eq!(decode_e4m3fn(encode_e4m3fn(1.9375)), 2.0);
        assert_eq!(decode_e4m3fn(encode_e4m3fn(248.0)), 256.0);
        assert_eq!(decode_e4m3fn(encode_e4m3fn(432.0)), 448.0);
    }
}
