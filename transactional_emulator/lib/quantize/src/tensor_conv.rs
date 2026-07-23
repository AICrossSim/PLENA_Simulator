//! Small `tch::Tensor` <-> `f32` conversion helpers shared across the emulator.
//!
//! These centralize two patterns that were previously copy-pasted into several
//! modules (dma, vector_machine, and the SRAM banks):
//!
//! - building an owned 1-D f32 tensor from a slice without tripping over
//!   `Tensor::from_slice`'s empty-slice edge case, and
//! - extracting a contiguous f32 `Vec` from an arbitrary-kind tensor safely
//!   (no `from_raw_parts` on a possibly non-f32 / non-contiguous buffer).

use tch::Tensor;

/// Build an owned, contiguous 1-D f32 tensor from `data`.
///
/// Returns an empty f32 tensor for an empty slice (matching `[0]`-shaped
/// zeros) instead of relying on `Tensor::from_slice`, which is awkward for
/// zero-length inputs. The returned tensor owns its storage.
pub fn tensor_from_f32_slice(data: &[f32]) -> Tensor {
    if data.is_empty() {
        return Tensor::zeros([0], (tch::Kind::Float, tch::Device::Cpu));
    }
    unsafe {
        Tensor::from_blob(
            data.as_ptr() as *const u8,
            &[data.len() as i64],
            &[],
            tch::Kind::Float,
            tch::Device::Cpu,
        )
        .internal_to_copy((tch::Kind::Float, tch::Device::Cpu), false)
    }
}

/// Copy a 1-D tensor into an f32 `Vec`.
///
/// The tensor is first cast to f32 and made contiguous, so this is safe for any
/// input dtype/layout. Falls back to element-wise reads if the bulk copy fails.
pub fn tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
    let len = tensor.size1().unwrap() as usize;
    let tensor_f32 = tensor.to_kind(tch::Kind::Float).contiguous();
    let mut data = vec![0.0f32; len];
    if tensor_f32.f_copy_data(&mut data, len).is_ok() {
        return data;
    }
    for (idx, value) in data.iter_mut().enumerate() {
        *value = tensor_f32.double_value(&[idx as i64]) as f32;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_handles_empty() {
        let t = tensor_from_f32_slice(&[]);
        assert_eq!(t.size1().unwrap(), 0);
    }

    #[test]
    fn round_trips_values() {
        let vals = [1.0f32, -2.5, 3.25, 0.0];
        let t = tensor_from_f32_slice(&vals);
        assert_eq!(tensor_to_f32_vec(&t), vals);
    }

    #[test]
    fn to_f32_vec_casts_non_float_kind() {
        let t = Tensor::from_slice(&[1i64, 2, 3]);
        assert_eq!(tensor_to_f32_vec(&t), vec![1.0, 2.0, 3.0]);
    }
}