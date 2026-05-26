#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpType {
    pub sign: bool,
    pub exponent: u8,
    pub mantissa: u8,
}

const fn mask(x: u8) -> u32 {
    ((1u64 << x) - 1) as _
}

/// Count leading zeros in an n-bit value (not the full 32-bit value)
const fn clz_n(val: u32, n: u8) -> u8 {
    if val == 0 {
        n
    } else {
        (n as u32 - (32 - val.leading_zeros())) as u8
    }
}

impl FpType {
    pub const E8M0: Self = FpType {
        sign: false,
        exponent: 8,
        mantissa: 0,
    };

    pub const F16: Self = FpType {
        sign: true,
        exponent: 5,
        mantissa: 10,
    };

    pub const BF16: Self = FpType {
        sign: true,
        exponent: 8,
        mantissa: 7,
    };

    pub const F32: Self = FpType {
        sign: true,
        exponent: 8,
        mantissa: 23,
    };

    pub const fn size_in_bits(self) -> u8 {
        self.sign as u8 + self.exponent + self.mantissa
    }

    pub const fn cast(self, new_ty: FpType, bits: u32) -> u32 {
        let sign = if self.sign {
            (bits >> (self.exponent + self.mantissa)) & 1
        } else {
            0
        };

        // Sign bit not representable, round to smallest representable number, i.e. zero.
        if sign == 1 && !new_ty.sign {
            return 0;
        }

        let mantissa_bits = bits & mask(self.mantissa);
        let exponent_mask = mask(self.exponent);
        let exponent = (bits >> self.mantissa) & exponent_mask;

        let new_exponent_mask = mask(new_ty.exponent);

        // For subnormal source when converting to larger format, we need to normalize
        // and convert to a normal number in the dest format.
        // E.g., E4M3 subnormal 0x07 = (7/8) * 2^(-7) = 0.0068359375 should become F32 normal.
        let (mut converted_exponent, subnormal_normalize_shift) = match exponent {
            // Subnormal handling
            0 => {
                if mantissa_bits == 0 {
                    // Zero stays zero
                    (0, 0)
                } else if self.exponent < new_ty.exponent {
                    // Source subnormal can become dest normal
                    // Subnormal value = (mantissa / 2^m) * 2^(-src_bias)
                    // Need to normalize: find leading 1 in mantissa and shift
                    let leading_zeros = clz_n(mantissa_bits, self.mantissa);
                    let normalize_shift = leading_zeros + 1; // +1 to make leading 1 implicit

                    // Source effective exponent for subnormal: -src_bias
                    // Note: Python quantizer uses -bias (not IEEE's 1-bias) for subnormals
                    let src_bias = (exponent_mask >> 1) as i32;
                    let effective_src_exp = -src_bias; // E4M3: -7 (matching Python quantizer)

                    // After normalization, exponent decreases
                    let normalized_exp = effective_src_exp - (normalize_shift as i32);

                    // Convert to dest biased exponent
                    let dst_bias = (new_exponent_mask >> 1) as i32;
                    let dst_exp = normalized_exp + dst_bias;

                    if dst_exp <= 0 {
                        // Underflow to dest subnormal - keep as subnormal
                        (0, 0)
                    } else {
                        (dst_exp as u32, normalize_shift)
                    }
                } else {
                    // Source and dest have same or dest has smaller exponent range
                    // Subnormal stays subnormal
                    (0, 0)
                }
            }
            // Inf/NaN -> Inf/NaN
            _ if exponent == exponent_mask => (new_exponent_mask, 0),
            // Normal number bias conversion
            _ if self.exponent <= new_ty.exponent => {
                (exponent + ((new_exponent_mask - exponent_mask) >> 1), 0)
            }
            _ => {
                // TODO: Needs to reimplment the underflow and overflow treatment.
                let bias_diff = (exponent - new_exponent_mask) >> 1;
                if exponent <= bias_diff {
                    // Underflow: saturate to zero (subnormal)
                    (0, 0)
                } else if exponent - bias_diff >= new_exponent_mask {
                    // Overflow: saturate to infinity
                    (new_exponent_mask, 0)
                } else {
                    (exponent - bias_diff, 0)
                }
            }
        };

        // For subnormal normalization, we need to shift the mantissa to remove the leading 1
        // that becomes implicit in the normalized representation.
        let normalized_mantissa = if subnormal_normalize_shift > 0 {
            // Shift left to normalize, masking off the now-implicit leading 1
            (mantissa_bits << subnormal_normalize_shift) & mask(self.mantissa)
        } else {
            mantissa_bits
        };

        let converted_mantissa = if self.mantissa <= new_ty.mantissa {
            normalized_mantissa << (new_ty.mantissa - self.mantissa)
        } else {
            // In this case, the conversion is lossy, we need to perform rounding.
            let discarded_bits = (mantissa_bits & mask(self.mantissa - new_ty.mantissa - 1)) != 0;
            let prelim_shift = mantissa_bits >> (self.mantissa - new_ty.mantissa - 1);
            let round_dir = match (prelim_shift & 3, discarded_bits) {
                // < 0.5, Round down
                (0b00 | 0b10, _) => 0,
                // > 0.5, Round up
                (0b01 | 0b11, true) => 1,
                // = 0.5, Round to even
                (0b01, false) => 0,
                (0b11, false) => 1,
                _ => unreachable!(),
            };
            let shift = (prelim_shift + round_dir) >> 1;
            if shift >> new_ty.mantissa != 0 {
                // Rounding overflow: increment exponent and zero mantissa (saturate to Inf on overflow)
                if converted_exponent < new_exponent_mask {
                    converted_exponent += 1;
                }
                // Saturate to Inf if exponent overflowed
                if converted_exponent >= new_exponent_mask {
                    converted_exponent = new_exponent_mask;
                }
                0
            } else {
                shift
            }
        };

        sign << (new_ty.exponent + new_ty.mantissa)
            | converted_exponent << new_ty.mantissa
            | converted_mantissa
    }

    /// Convert f32 to bits. The conversion is lossy and is by rounding.
    pub const fn bits_from_f32(self, float: f32) -> u32 {
        Self::F32.cast(self, float.to_bits())
    }

    /// Convert bits to f32. Only lower `bits()` bits are used.
    pub const fn convert_bits_to_f32(self, bits: u32) -> f32 {
        f32::from_bits(self.cast(Self::F32, bits))
    }
}

#[test]
fn test_f32() {
    let ty = FpType::F32;

    assert_eq!(ty.convert_bits_to_f32(0f32.to_bits()), 0f32);
    assert_eq!(ty.convert_bits_to_f32(1f32.to_bits()), 1f32);
    assert_eq!(
        ty.convert_bits_to_f32(f32::INFINITY.to_bits()),
        f32::INFINITY
    );
    assert_eq!(
        ty.convert_bits_to_f32(f32::NEG_INFINITY.to_bits()),
        f32::NEG_INFINITY
    );
}

#[test]
fn test_f16() {
    use half::f16;

    let ty = FpType::F16;

    assert_eq!(ty.convert_bits_to_f32(f16::ZERO.to_bits() as u32), 0f32);
    assert_eq!(ty.convert_bits_to_f32(f16::ONE.to_bits() as u32), 1f32);
    assert_eq!(
        ty.convert_bits_to_f32(f16::INFINITY.to_bits() as u32),
        f32::INFINITY
    );
    assert_eq!(
        ty.convert_bits_to_f32(f16::NEG_INFINITY.to_bits() as u32),
        f32::NEG_INFINITY
    );
}

#[test]
fn test_e4m3_subnormal() {
    // E4M3 format: 1 sign, 4 exp, 3 mantissa. Bias = 7.
    let ty = FpType {
        sign: true,
        exponent: 4,
        mantissa: 3,
    };

    // Test subnormal values (exponent = 0)
    // E4M3 subnormal: value = (mantissa / 8) * 2^(-7) (matching Python quantizer convention)
    // Note: IEEE standard uses 2^(1-bias)=2^(-6), but Python uses 2^(-bias)=2^(-7)

    // 0x07 = exp=0, man=7 -> (7/8) * 2^(-7) = 0.875 * 0.0078125 = 0.0068359375
    let val = ty.convert_bits_to_f32(0x07);
    assert!((val - 0.0068359375).abs() < 1e-9, "E4M3 subnormal 0x07: got {}, expected 0.0068359375", val);

    // 0x87 = sign=1, exp=0, man=7 -> -0.0068359375
    let val = ty.convert_bits_to_f32(0x87);
    assert!((val - (-0.0068359375)).abs() < 1e-9, "E4M3 subnormal 0x87: got {}, expected -0.0068359375", val);

    // 0x01 = exp=0, man=1 -> (1/8) * 2^(-7) = 0.0009765625
    let val = ty.convert_bits_to_f32(0x01);
    assert!((val - 0.0009765625).abs() < 1e-9, "E4M3 subnormal 0x01: got {}, expected 0.0009765625", val);

    // 0x04 = exp=0, man=4 -> (4/8) * 2^(-7) = 0.5 * 0.0078125 = 0.00390625
    let val = ty.convert_bits_to_f32(0x04);
    assert!((val - 0.00390625).abs() < 1e-9, "E4M3 subnormal 0x04: got {}, expected 0.00390625", val);

    // 0x00 = zero
    assert_eq!(ty.convert_bits_to_f32(0x00), 0.0);

    // Test some normal values for sanity
    // 0x38 = exp=7, man=0 -> 1.0 * 2^(7-7) = 1.0
    let val = ty.convert_bits_to_f32(0x38);
    assert!((val - 1.0).abs() < 1e-6, "E4M3 normal 0x38: got {}, expected 1.0", val);

    // 0x3F = exp=7, man=7 -> 1.875 * 2^(7-7) = 1.875
    let val = ty.convert_bits_to_f32(0x3F);
    assert!((val - 1.875).abs() < 1e-6, "E4M3 normal 0x3F: got {}, expected 1.875", val);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntType {
    pub width: u32,
}

impl IntType {
    pub const fn size_in_bits(self) -> u8 {
        self.width as u8
    }

    /// Convert f32 to integer bits. Truncates the float to an integer.
    pub const fn bits_from_f32(self, float: f32) -> u32 {
        let int_val = float as i32;
        let mask = if self.width >= 32 {
            0xFFFFFFFFu32
        } else {
            ((1u64 << self.width) - 1) as u32
        };
        (int_val as u32) & mask
    }

    /// Convert integer bits to f32. Interprets bits as unsigned integer.
    pub const fn convert_bits_to_f32(self, bits: u32) -> f32 {
        let mask = if self.width >= 32 {
            0xFFFFFFFFu32
        } else {
            ((1u64 << self.width) - 1) as u32
        };
        let masked_bits = bits & mask;
        masked_bits as f32
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Fp(FpType),
    Int(IntType),
}

impl From<FpType> for DataType {
    fn from(value: FpType) -> Self {
        Self::Fp(value)
    }
}



impl DataType {
    pub fn size_in_bits(self) -> u8 {
        match self {
            DataType::Fp(fp_type) => fp_type.size_in_bits(),
            DataType::Int(int_type) => int_type.size_in_bits(),
        }
    }

    pub const fn bits_from_f32(self, float: f32) -> u32 {
        match self {
            DataType::Fp(fp_type) => fp_type.bits_from_f32(float),
            DataType::Int(int_type) => int_type.bits_from_f32(float),
        }
    }

    pub const fn convert_bits_to_f32(self, bits: u32) -> f32 {
        match self {
            DataType::Fp(fp_type) => fp_type.convert_bits_to_f32(bits),
            DataType::Int(int_type) => int_type.convert_bits_to_f32(bits),
        }
    }

    /// Convert bytes to vector of f32.
    pub fn convert_bytes_to_f32_vec(self, mut bytes: &[u8], out: &mut [f32]) {
        let bits = self.size_in_bits();
        let mut data = 0;
        let mut bits_left = 0;
        for out in out.iter_mut() {
            while bits_left < bits {
                data |= (bytes[0] as u32) << bits_left;
                bits_left += 8;
                bytes = &bytes[1..];
            }

            *out = self.convert_bits_to_f32(data);
            bits_left -= bits;
            data >>= bits;
        }
    }

    pub fn bytes_from_f32(self, input: &[f32], mut out: &mut [u8]) {
        let bits = self.size_in_bits();
        let mut data = 0;
        let mut bits_left = 0u8;

        for elem in input.iter().copied() {
            while bits_left >= 8 {
                out[0] = data as u8;
                out = &mut out[1..];
                data >>= 8;
                bits_left -= 8;
            }

            data |= self.bits_from_f32(elem) << bits_left;
            bits_left += bits;
        }

        while bits_left > 0 {
            out[0] = data as u8;
            out = &mut out[1..];
            data >>= 8;
            bits_left = bits_left.saturating_sub(8);
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        let size = self.size_in_bits();
        assert!(size.is_multiple_of(8));
        size as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxDataType {
    Plain(DataType),
    Mx {
        elem: DataType,
        scale: DataType,
        block: u32,
    },
}

impl MxDataType {
    pub fn element_type(self) -> DataType {
        match self {
            MxDataType::Plain(elem) => elem,
            MxDataType::Mx { elem, .. } => elem,
        }
    }

    /// Returns the size in bits of the element type
    /// Works for both Plain (FP and Int) and Mx variants
    pub fn size_in_bits(self) -> u8 {
        match self {
            MxDataType::Plain(data_type) => data_type.size_in_bits(),
            MxDataType::Mx { elem, .. } => elem.size_in_bits(),
        }
    }
}

impl From<FpType> for MxDataType {
    fn from(value: FpType) -> Self {
        MxDataType::Plain(value.into())
    }
}

impl From<DataType> for MxDataType {
    fn from(value: DataType) -> Self {
        MxDataType::Plain(value)
    }
}
