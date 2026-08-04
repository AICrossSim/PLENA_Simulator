//! Pure functional Nemotron Mamba-2 mixer.
//!
//! This kernel has no ISA, HBM, queue, or timing behavior. Keeping the
//! numerical path independent lets differential tests isolate precision and
//! recurrence bugs before command dispatch is enabled.

use half::bf16;

use super::{MambaDescriptor, MambaError, PrecisionPolicy};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FunctionalMambaWeights {
    pub(crate) in_proj_weight: Vec<f32>,
    pub(crate) in_proj_bias: Option<Vec<f32>>,
    pub(crate) conv_weight: Vec<f32>,
    pub(crate) conv_bias: Option<Vec<f32>>,
    pub(crate) a_log: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) d_skip: Vec<f32>,
    pub(crate) norm_weight: Vec<f32>,
    pub(crate) out_proj_weight: Vec<f32>,
    pub(crate) out_proj_bias: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FunctionalMambaState {
    /// `[batch, head, head_dim, state_dim]`, row-major FP32.
    pub(crate) ssm: Vec<f32>,
    /// `[batch, conv_channel, conv_kernel]`, oldest sample first, FP32.
    pub(crate) conv: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FunctionalMambaResult {
    /// `[batch, sequence, d_model]`, represented as FP32 values at the selected
    /// external precision boundary.
    pub(crate) output: Vec<f32>,
    pub(crate) state: FunctionalMambaState,
}

fn to_usize(value: u64) -> Result<usize, MambaError> {
    value
        .try_into()
        .map_err(|_| MambaError::UnsupportedProfile("tensor exceeds host address space"))
}

fn usize_product(values: &[u64]) -> Result<usize, MambaError> {
    let value = values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(MambaError::UnsupportedProfile(
                "functional tensor size overflows u64",
            ))
    })?;
    to_usize(value)
}

fn boundary(policy: PrecisionPolicy, value: f32) -> f32 {
    match policy {
        PrecisionPolicy::Fp32Reference => value,
        PrecisionPolicy::Bf16ActivationFp32State => f32::from(bf16::from_f32(value)),
    }
}

fn silu(value: f32) -> f32 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        value * exp / (1.0 + exp)
    }
}

fn softplus(value: f32) -> f32 {
    // Match torch.nn.functional.softplus's default threshold.
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn require_len(actual: usize, expected: usize) -> Result<(), MambaError> {
    if actual != expected {
        return Err(MambaError::Internal(
            "functional tensor length does not match descriptor",
        ));
    }
    Ok(())
}

impl FunctionalMambaWeights {
    fn validate(&self, descriptor: &MambaDescriptor) -> Result<(), MambaError> {
        let d_model = descriptor.d_model as u64;
        let d_inner = descriptor.d_inner as u64;
        let heads = descriptor.num_heads as u64;
        let conv_channels = descriptor.conv_channels()?;
        let projection = descriptor.projection_size()?;
        let output_projection = descriptor.output_projection_size()?;
        require_len(
            self.in_proj_weight.len(),
            usize_product(&[projection, d_model])?,
        )?;
        if let Some(bias) = &self.in_proj_bias {
            require_len(bias.len(), to_usize(projection)?)?;
        }
        require_len(
            self.conv_weight.len(),
            usize_product(&[conv_channels, descriptor.conv_kernel as u64])?,
        )?;
        if let Some(bias) = &self.conv_bias {
            require_len(bias.len(), to_usize(conv_channels)?)?;
        }
        require_len(self.a_log.len(), to_usize(heads)?)?;
        require_len(self.dt_bias.len(), to_usize(heads)?)?;
        require_len(self.d_skip.len(), to_usize(heads)?)?;
        require_len(self.norm_weight.len(), to_usize(d_inner)?)?;
        require_len(
            self.out_proj_weight.len(),
            usize_product(&[d_model, output_projection])?,
        )?;
        if let Some(bias) = &self.out_proj_bias {
            require_len(bias.len(), to_usize(d_model)?)?;
        }
        Ok(())
    }
}

impl FunctionalMambaState {
    pub(crate) fn zeros(descriptor: &MambaDescriptor) -> Result<Self, MambaError> {
        Ok(Self {
            ssm: vec![
                0.0;
                usize_product(&[
                    descriptor.batch_size as u64,
                    descriptor.num_heads as u64,
                    descriptor.head_dim as u64,
                    descriptor.state_dim as u64,
                ])?
            ],
            conv: vec![
                0.0;
                usize_product(&[
                    descriptor.batch_size as u64,
                    descriptor.conv_channels()?,
                    descriptor.conv_kernel as u64,
                ])?
            ],
        })
    }

    fn validate(&self, descriptor: &MambaDescriptor) -> Result<(), MambaError> {
        let expected = Self::zeros(descriptor)?;
        require_len(self.ssm.len(), expected.ssm.len())?;
        require_len(self.conv.len(), expected.conv.len())?;
        Ok(())
    }
}

fn linear_row(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output_features: usize,
    policy: PrecisionPolicy,
) -> Vec<f32> {
    let input_features = input.len();
    let mut output = vec![0.0; output_features];
    for row in 0..output_features {
        let weights = &weight[row * input_features..(row + 1) * input_features];
        let mut sum = 0.0f32;
        for index in 0..input_features {
            sum = boundary(policy, input[index]).mul_add(boundary(policy, weights[index]), sum);
        }
        if let Some(bias) = bias {
            sum += boundary(policy, bias[row]);
        }
        output[row] = boundary(policy, sum);
    }
    output
}

pub(crate) fn execute(
    descriptor: &MambaDescriptor,
    input: &[f32],
    weights: &FunctionalMambaWeights,
    initial_state: Option<&FunctionalMambaState>,
) -> Result<FunctionalMambaResult, MambaError> {
    weights.validate(descriptor)?;
    let batch = descriptor.batch_size as usize;
    let sequence = descriptor.sequence_length as usize;
    let d_model = descriptor.d_model as usize;
    let d_inner = descriptor.d_inner as usize;
    let heads = descriptor.num_heads as usize;
    let head_dim = descriptor.head_dim as usize;
    let state_dim = descriptor.state_dim as usize;
    let groups = descriptor.groups as usize;
    let conv_kernel = descriptor.conv_kernel as usize;
    let conv_channels = to_usize(descriptor.conv_channels()?)?;
    let projection = to_usize(descriptor.projection_size()?)?;
    let d_mlp = descriptor.d_mlp as usize;
    let output_projection = to_usize(descriptor.output_projection_size()?)?;
    require_len(
        input.len(),
        usize_product(&[batch as u64, sequence as u64, d_model as u64])?,
    )?;

    let mut state = if descriptor.continue_state() {
        let state = initial_state.ok_or(MambaError::Internal(
            "continuing command has no functional initial state",
        ))?;
        state.validate(descriptor)?;
        state.clone()
    } else {
        FunctionalMambaState::zeros(descriptor)?
    };
    let mut output = vec![0.0; input.len()];
    let heads_per_group = heads / groups;
    let group_width = d_inner / groups;

    for batch_index in 0..batch {
        for token in 0..sequence {
            let input_start = (batch_index * sequence + token) * d_model;
            let projected = linear_row(
                &input[input_start..input_start + d_model],
                &weights.in_proj_weight,
                weights.in_proj_bias.as_deref(),
                projection,
                descriptor.precision,
            );
            let gate_start = 2 * d_mlp;
            let xbc_start = gate_start + d_inner;
            let dt_start = xbc_start + conv_channels;
            let gate = &projected[gate_start..xbc_start];
            let xbc = &projected[xbc_start..dt_start];
            let dt_raw = &projected[dt_start..dt_start + heads];

            let mut conv_output = vec![0.0f32; conv_channels];
            for channel in 0..conv_channels {
                let state_start = (batch_index * conv_channels + channel) * conv_kernel;
                state
                    .conv
                    .copy_within(state_start + 1..state_start + conv_kernel, state_start);
                state.conv[state_start + conv_kernel - 1] =
                    boundary(descriptor.precision, xbc[channel]);
                let weight_start = channel * conv_kernel;
                let mut sum = 0.0f32;
                for kernel_index in 0..conv_kernel {
                    sum += state.conv[state_start + kernel_index]
                        * boundary(
                            descriptor.precision,
                            weights.conv_weight[weight_start + kernel_index],
                        );
                }
                if let Some(bias) = &weights.conv_bias {
                    sum += boundary(descriptor.precision, bias[channel]);
                }
                conv_output[channel] = boundary(descriptor.precision, silu(sum));
            }

            let b_start = d_inner;
            let c_start = b_start + groups * state_dim;
            let mut scan_output = vec![0.0f32; d_inner];
            for head in 0..heads {
                let dt =
                    softplus(dt_raw[head] + boundary(descriptor.precision, weights.dt_bias[head]))
                        .clamp(descriptor.dt_min, descriptor.dt_max);
                let a = -boundary(descriptor.precision, weights.a_log[head]).exp();
                let decay = (dt * a).exp();
                let group = head / heads_per_group;
                for position in 0..head_dim {
                    let inner_index = head * head_dim + position;
                    let x = conv_output[inner_index];
                    let mut value = 0.0f32;
                    for state_index in 0..state_dim {
                        let b = conv_output[b_start + group * state_dim + state_index];
                        let c = conv_output[c_start + group * state_dim + state_index];
                        let index = (((batch_index * heads + head) * head_dim + position)
                            * state_dim)
                            + state_index;
                        let drive = (dt * b) * x;
                        let next = state.ssm[index] * decay + drive;
                        state.ssm[index] = next;
                        value += next * c;
                    }
                    scan_output[inner_index] =
                        value + boundary(descriptor.precision, weights.d_skip[head]) * x;
                }
            }

            let mut normalized = vec![0.0f32; d_inner];
            for group in 0..groups {
                let start = group * group_width;
                let mut gated = vec![0.0f32; group_width];
                let mut square_sum = 0.0f32;
                for offset in 0..group_width {
                    let index = start + offset;
                    let value =
                        scan_output[index] * silu(boundary(descriptor.precision, gate[index]));
                    gated[offset] = value;
                    square_sum += value * value;
                }
                let inverse_rms = (square_sum / group_width as f32 + descriptor.rms_norm_eps)
                    .sqrt()
                    .recip();
                for offset in 0..group_width {
                    let index = start + offset;
                    normalized[index] = boundary(
                        descriptor.precision,
                        gated[offset]
                            * inverse_rms
                            * boundary(descriptor.precision, weights.norm_weight[index]),
                    );
                }
            }

            let mut output_input = Vec::with_capacity(output_projection);
            for index in 0..d_mlp {
                output_input.push(boundary(
                    descriptor.precision,
                    silu(projected[index]) * projected[d_mlp + index],
                ));
            }
            output_input.extend_from_slice(&normalized);
            let output_row = linear_row(
                &output_input,
                &weights.out_proj_weight,
                weights.out_proj_bias.as_deref(),
                d_model,
                descriptor.precision,
            );
            let output_start = (batch_index * sequence + token) * d_model;
            output[output_start..output_start + d_model].copy_from_slice(&output_row);
        }
    }

    Ok(FunctionalMambaResult { output, state })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        policy: PrecisionPolicy,
        sequence_length: u32,
        continuing: bool,
    ) -> MambaDescriptor {
        MambaDescriptor {
            flags: u32::from(continuing),
            context_id: 1,
            sequence_length,
            batch_size: 1,
            d_model: 4,
            d_inner: 4,
            num_heads: 2,
            head_dim: 2,
            state_dim: 2,
            groups: 1,
            chunk_size: 2,
            conv_kernel: 2,
            precision: policy,
            rms_norm_eps: 1e-5,
            input_addr: 0,
            output_addr: 0,
            in_proj_weight_addr: 0,
            in_proj_bias_addr: 0,
            conv_weight_addr: 0,
            conv_bias_addr: 0,
            a_log_addr: 0,
            d_skip_addr: 0,
            norm_weight_addr: 0,
            out_proj_weight_addr: 0,
            out_proj_bias_addr: 0,
            ssm_state_addr: 0,
            conv_state_addr: 0,
            scratch_addr: 0,
            completion_addr: 0,
            dt_bias_addr: 0,
            input_batch_stride: 0,
            input_token_stride: 0,
            output_batch_stride: 0,
            output_token_stride: 0,
            state_request_stride: 64,
            state_head_stride: 32,
            conv_request_stride: 64,
            scratch_bytes: 0,
            dependency_event: u32::MAX,
            completion_event: u32::MAX,
            layer_id: 0,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
            d_mlp: 0,
        }
    }

    fn values(length: usize, scale: f32, offset: i32) -> Vec<f32> {
        (0..length)
            .map(|index| (((index as i32 * 17 + offset) % 23) - 11) as f32 * scale)
            .collect()
    }

    fn weights() -> FunctionalMambaWeights {
        FunctionalMambaWeights {
            in_proj_weight: values(14 * 4, 0.015, 3),
            in_proj_bias: Some(values(14, 0.01, 5)),
            conv_weight: values(8 * 2, 0.025, 7),
            conv_bias: Some(values(8, 0.01, 11)),
            a_log: values(2, 0.02, 13),
            dt_bias: values(2, 0.03, 17),
            d_skip: values(2, 0.04, 19),
            norm_weight: vec![0.8, 0.9, 1.0, 1.1],
            out_proj_weight: values(4 * 4, 0.02, 2),
            out_proj_bias: Some(values(4, 0.01, 4)),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "value {index}: actual={actual:?}, expected={expected:?}, error={error:?}, tolerance={tolerance:?}"
            );
        }
    }

    fn assert_bf16_ulp(actual: &[f32], expected: &[f32], max_ulps: u16) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual_bits = bf16::from_f32(*actual).to_bits();
            let expected_bits = bf16::from_f32(*expected).to_bits();
            assert_eq!(
                actual_bits >> 15,
                expected_bits >> 15,
                "value {index} has different BF16 signs: actual={actual:?}, expected={expected:?}"
            );
            let distance = actual_bits.abs_diff(expected_bits);
            assert!(
                distance <= max_ulps,
                "value {index}: actual={actual:?}, expected={expected:?}, BF16 ULP distance={distance}"
            );
        }
    }

    #[test]
    fn prefill_matches_separate_prefill_and_step_commands() {
        let input = values(3 * 4, 0.05, 1);
        for policy in [
            PrecisionPolicy::Fp32Reference,
            PrecisionPolicy::Bf16ActivationFp32State,
        ] {
            let full = execute(&descriptor(policy, 3, false), &input, &weights(), None).unwrap();

            let first =
                execute(&descriptor(policy, 1, false), &input[..4], &weights(), None).unwrap();
            let second = execute(
                &descriptor(policy, 1, true),
                &input[4..8],
                &weights(),
                Some(&first.state),
            )
            .unwrap();
            let third = execute(
                &descriptor(policy, 1, true),
                &input[8..12],
                &weights(),
                Some(&second.state),
            )
            .unwrap();
            let mut stepped_output = first.output;
            stepped_output.extend_from_slice(&second.output);
            stepped_output.extend_from_slice(&third.output);

            assert_eq!(full.output, stepped_output);
            assert_eq!(full.state, third.state);
        }
    }

    #[test]
    fn matches_pytorch_golden_for_fp32_and_bf16_boundaries() {
        let input = values(3 * 4, 0.05, 1);
        let fp32 = execute(
            &descriptor(PrecisionPolicy::Fp32Reference, 3, false),
            &input,
            &weights(),
            None,
        )
        .unwrap();
        assert_close(
            &fp32.output,
            &[
                -0.12705076,
                0.04535317,
                -0.012242902,
                -0.05531925,
                -0.11856744,
                0.052201636,
                -0.007029295,
                -0.04508653,
                -0.08400881,
                0.08750807,
                0.029024947,
                -0.02340853,
            ],
            2e-6,
        );
        assert_close(
            &fp32.state.ssm,
            &[
                -0.00011653046,
                -0.0001659228,
                0.00003939421,
                0.00091713906,
                -0.0002043631,
                -0.0017785144,
                -0.000041582047,
                -0.0010278731,
            ],
            2e-7,
        );
        assert_close(
            &fp32.state.conv,
            &[
                -0.10675001,
                0.077,
                0.071499996,
                0.241,
                0.019749999,
                0.17500001,
                -0.031999998,
                0.109,
                -0.08374999,
                0.043000005,
                -0.009000003,
                0.08625001,
                -0.060750008,
                0.02025,
                -0.112500004,
                -0.04575,
            ],
            2e-7,
        );

        let bf16 = execute(
            &descriptor(PrecisionPolicy::Bf16ActivationFp32State, 3, false),
            &input,
            &weights(),
            None,
        )
        .unwrap();
        assert_bf16_ulp(
            &bf16.output,
            &[
                -0.12695313,
                0.045654297,
                -0.012084961,
                -0.05517578,
                -0.118652344,
                0.052246094,
                -0.007080078,
                -0.045166016,
                -0.083984375,
                0.087890625,
                0.029052734,
                -0.02331543,
            ],
            2,
        );
        assert_close(
            &bf16.state.ssm,
            &[
                -0.00011694061,
                -0.00016479181,
                0.000038569327,
                0.00091674295,
                -0.00020349291,
                -0.00176965,
                -0.000041611733,
                -0.0010249581,
            ],
            2e-7,
        );
        assert_close(
            &bf16.state.conv,
            &[
                -0.10644531,
                0.07714844,
                0.07128906,
                0.24121094,
                0.019897461,
                0.17480469,
                -0.031982422,
                0.109375,
                -0.083984375,
                0.04296875,
                -0.009277344,
                0.0859375,
                -0.061035156,
                0.020385742,
                -0.11279297,
                -0.045654297,
            ],
            0.0,
        );
    }
}
