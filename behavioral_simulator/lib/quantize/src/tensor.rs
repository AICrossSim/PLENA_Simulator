use anyhow::Result;
use tch::Tensor;

use crate::dtype::{DataType, FpType, MxDataType};

/// Quantize a single FP32 value to minifloat format using IEEE hardware quantization
/// This matches the Python _minifloat_ieee_quantize_hardware function
fn minifloat_ieee_quantize_hardware(value: f32, fp_type: FpType) -> u32 {
    if value == 0.0 {
        return 0;
    }
    
    let width = fp_type.size_in_bits();
    let exponent_width = fp_type.exponent;
    let mantissa_bits = width - exponent_width - (if fp_type.sign { 1 } else { 0 });
    
    // Default bias: 2^(exponent_width - 1) - 1
    let exponent_bias = (1u32 << (exponent_width - 1)) - 1;
    let exponent_max = (1u32 << exponent_width) - 2 - exponent_bias;
    let exponent_min = -(exponent_bias as i32);
    
    let shifted_mantissa_max = (1u32 << mantissa_bits) - 1;
    let shifted_mantissa_min = 0u32;
    
    // Extract sign
    let sign = if fp_type.sign && value < 0.0 { 1u32 } else { 0u32 };
    let abs_value = value.abs();
    
    // Calculate exponent: floor(log2(value + 1e-9))
    let epsilon = 1e-9;
    let raw_exp = (abs_value + epsilon).log2().floor() as i32;
    let overflow = raw_exp > exponent_max as i32;
    
    // Clamp exponent
    let exponent = raw_exp.max(exponent_min).min(exponent_max as i32);
    
    // Calculate mantissa
    let mantissa = abs_value / 2.0f32.powi(exponent);
    
    // Check if normal (exponent != -exponent_bias)
    let is_normal = exponent != -(exponent_bias as i32);
    
    // Quantize mantissa
    let shift = 1u32 << mantissa_bits;
    let shifted_mantissa = if is_normal {
        // Normal: (mantissa - 1) * shift, round, clamp
        let shifted = ((mantissa - 1.0) * shift as f32).round() as u32;
        shifted.max(shifted_mantissa_min).min(shifted_mantissa_max)
    } else {
        // Subnormal: mantissa * shift, round, clamp
        let shifted = (mantissa * shift as f32).round() as u32;
        shifted.max(shifted_mantissa_min).min(shifted_mantissa_max)
    };
    
    // Handle overflow: saturate to max mantissa
    let shifted_mantissa = if overflow {
        shifted_mantissa_max
    } else {
        shifted_mantissa
    };
    
    // Encode the minifloat bits
    // Format: [sign][exponent][mantissa]
    let biased_exponent = (exponent + exponent_bias as i32) as u32;
    let exponent_mask = ((1u32 << exponent_width) - 1) << mantissa_bits;
    let mantissa_mask = (1u32 << mantissa_bits) - 1;
    
    (sign << (exponent_width + mantissa_bits)) | (biased_exponent << mantissa_bits) | (shifted_mantissa & mantissa_mask)
}

#[cfg(test)]
mod tensor_test;

pub struct QuantTensor {
    tensor: Tensor,
    ty: MxDataType,
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
        Ok(QuantTensor { tensor: tensor, ty })
    }

    /// Create a quantized tensor, assuming the tensor is already quantized.
    pub fn quantize(tensor: Tensor, ty: MxDataType) -> Self {
        // TODO: add actual quantization
        Self::new_assuming_quantized(tensor, ty).unwrap()
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
        let elem_ty = ty.element_type();

        let mut vec = vec![0f32; len];
        elem_ty.convert_bytes_to_f32_vec(bytes, &mut vec);

        if let MxDataType::Mx {
            elem: _,
            scale,
            block,
        } = ty
        {
            let mut scale_vec = vec![0f32; len / block as usize];

            scale.convert_bytes_to_f32_vec(&scale_bytes, &mut scale_vec);

            for (elem, scale) in vec
                .chunks_mut(block as usize)
                .zip(scale_vec.iter().copied())
            {
                for elem in elem.iter_mut() {
                    *elem *= scale;
                }
            }
        }

        let tensor = tch::Tensor::from_slice(&vec);
        Self { tensor, ty }
    }

    /// Serialize the quantized tensor into bytes.
    pub fn into_bytes(&mut self) -> (Vec<u8>, Vec<u8>) {
        let len = self.tensor.size1().unwrap() as usize;
        let slice =
            unsafe { core::slice::from_raw_parts(self.tensor.data_ptr() as *const f32, len) };
        println!("slice: {:?}", slice);

        let elem_ty = self.ty.element_type();
        
        if let MxDataType::Mx {
            elem,
            scale,
            block,
        } = self.ty
        {
            // Properly calculate MX scales and quantize elements
            let num_blocks = len / block as usize;
            let mut scale_vec = vec![0f32; num_blocks];
            let mut out = vec![0; len * elem.size_in_bits() as usize / 8];
            
            // Process each block
            for (block_idx, block_data) in slice.chunks(block as usize).enumerate() {
                if block_idx >= num_blocks {
                    break;
                }
                println!("block_idx: {}", block_idx);
                println!("block_data: {:?}", block_data);
                // Find maximum absolute value in this block
                let max_abs = block_data
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0f32, f32::max);
                
                if max_abs == 0.0 {
                    // All zeros: scale is 0 (will be encoded as 0)
                    scale_vec[block_idx] = 0.0;
                    // Elements are already zeros in the output
                } else {
                    // Calculate shared exponent bias following Python MXFP quantizer
                    // Python: per_block_exponent_bias = clamp(floor(log2(max)), -2^(bias_width-1), 2^(bias_width-1)-1)
                    // Extract FpType from DataType
                    let scale_exp_bits = match scale {
                        DataType::Fp(scale_fp) => scale_fp.exponent,
                        _ => 8, // Default
                    };
                    
                    // Calculate bias: 2^(exponent_bias_width - 1) - 1
                    // In MXFP, exponent_bias_width = scale_exp_bits
                    let bias_bias = (1u32 << (scale_exp_bits - 1)) - 1;
                    let scale_exp_min = -(bias_bias as i32);
                    let scale_exp_max = bias_bias as i32;
                    
                    // Calculate raw exponent from max_abs: floor(log2(max_abs))
                    // Add small epsilon to avoid log2(0) issues (Python adds 1e-9)
                    let max_abs_with_eps = max_abs + 1e-9;
                    let raw_exp = max_abs_with_eps.log2().floor() as i32;
                    
                    // Clamp to scale exponent range: [-2^(bias_width-1), 2^(bias_width-1)-1]
                    let per_block_exponent_bias = raw_exp.max(scale_exp_min).min(scale_exp_max);

                    // Calculate scale value: 2^per_block_exponent_bias (raw, not biased)
                    // This is the actual scale value used to divide elements
                    let scale_value = 2.0f32.powi(per_block_exponent_bias);
                    
                    // Store biased exponent for encoding: per_block_exponent_bias + bias_bias
                    let stored_scale = per_block_exponent_bias + bias_bias as i32;
                    
                    // Encode as float: create a float with FP32 exponent = stored_scale
                    // FP32 exponent is stored as (exp + 127), so value = 2^(stored_scale - 127)
                    let scale_encoded_value = 2.0f32.powi(stored_scale - 127);
                    scale_vec[block_idx] = scale_encoded_value;
                    
                    // Scale elements and quantize them
                    let scaled_elements: Vec<f32> = block_data
                        .iter()
                        .map(|&x| if x == 0.0 { 0.0 } else { x / scale_value })
                        .collect();

                    // Quantize each scaled element using proper minifloat IEEE quantization
                    let block_start_byte = block_idx * block as usize * elem.size_in_bits() as usize / 8;
                    let elem_bits = elem.size_in_bits() as usize;
                    let elem_bytes = (elem_bits + 7) / 8; // Round up for partial bytes
                    let block_bytes = block as usize * elem_bytes;
                    
                    // Extract FpType for quantization
                    if let DataType::Fp(elem_fp) = elem {
                        for (i, &scaled_val) in scaled_elements.iter().enumerate() {
                            let bits = minifloat_ieee_quantize_hardware(scaled_val, elem_fp);
                            let byte_offset = block_start_byte + i * elem_bytes;
                            // Write bits in little-endian order
                            for j in 0..elem_bytes {
                                if byte_offset + j < out.len() {
                                    out[byte_offset + j] = ((bits >> (j * 8)) & 0xFF) as u8;
                                }
                            }
                        }
                    } else {
                        // For non-FP types, fall back to bytes_from_f32
                        elem.bytes_from_f32(&scaled_elements, &mut out[block_start_byte..block_start_byte + block_bytes]);
                    }
                    // Print out this section of the 'out' buffer as hex bytes for easier inspection
                    let out_slice = &out[block_start_byte..(block_start_byte + block_bytes).min(out.len())];
                    print!("out[block {}] bytes: [", block_idx);
                    for (i, byte) in out_slice.iter().enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        print!("{:02x}", byte);
                    }
                    println!("]");
                }
            }
            
            // Convert scales to bytes
            let mut scale_out = vec![0; num_blocks * scale.size_in_bits() as usize / 8];
            scale.bytes_from_f32(&scale_vec, &mut scale_out);
            println!("scale_out: {:?}", scale_out);

            return (out, scale_out);
        }
        
        // Plain type: no scales
        let mut out = vec![0; len * elem_ty.size_in_bits() as usize / 8];
        elem_ty.bytes_from_f32(slice, &mut out);
        (out, Vec::new())
    }
}
