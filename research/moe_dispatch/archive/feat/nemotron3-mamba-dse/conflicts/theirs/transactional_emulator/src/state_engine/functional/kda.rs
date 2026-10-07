use std::sync::Arc;

use memory::ErasedMemoryModel;
use sram::VectorSram;

use super::{ParameterReader, sigmoid, silu};
use crate::state_engine::cache::StateBuffers;
use crate::state_engine::descriptor::{KdaPayload, StateDescriptor};
use crate::state_engine::error::StateEngineError;
use crate::state_engine::layout::LayoutStore;
use crate::state_engine::precision::{decode_tensor, requantize};
use crate::state_engine::projection::ProjectionBufferStats;
use crate::state_engine::vram::VramAccess;

struct KdaParameters {
    q_conv_weight: Vec<f32>,
    k_conv_weight: Vec<f32>,
    v_conv_weight: Vec<f32>,
    q_conv_bias: Option<Vec<f32>>,
    k_conv_bias: Option<Vec<f32>>,
    v_conv_bias: Option<Vec<f32>>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
}

pub async fn execute(
    hbm: &Arc<dyn ErasedMemoryModel>,
    vram: &Arc<VectorSram>,
    layouts: &mut LayoutStore,
    descriptor: &StateDescriptor,
    payload: &KdaPayload,
    buffers: &mut StateBuffers,
) -> Result<ProjectionBufferStats, StateEngineError> {
    let batch = usize::from(descriptor.batch_size);
    let heads = usize::from(descriptor.num_heads);
    let key_dim = usize::from(payload.key_dim);
    let value_dim = usize::from(payload.value_dim);
    let kernel = usize::from(payload.conv_kernel);
    let key_elements = heads * key_dim;
    let value_elements = heads * value_dim;
    let channels = 2 * key_elements + value_elements;

    let mut reader = ParameterReader::new(
        hbm,
        descriptor.parameter_precision,
        payload.parameter_scale_addr,
    );
    let parameters = KdaParameters {
        q_conv_weight: reader
            .required(
                "KDA q_conv_weight",
                payload.q_conv_weight_addr,
                key_elements * kernel,
                kernel,
            )
            .await?,
        k_conv_weight: reader
            .required(
                "KDA k_conv_weight",
                payload.k_conv_weight_addr,
                key_elements * kernel,
                kernel,
            )
            .await?,
        v_conv_weight: reader
            .required(
                "KDA v_conv_weight",
                payload.v_conv_weight_addr,
                value_elements * kernel,
                kernel,
            )
            .await?,
        q_conv_bias: reader
            .optional(payload.q_conv_bias_addr, key_elements, key_elements)
            .await?,
        k_conv_bias: reader
            .optional(payload.k_conv_bias_addr, key_elements, key_elements)
            .await?,
        v_conv_bias: reader
            .optional(payload.v_conv_bias_addr, value_elements, value_elements)
            .await?,
        a_log: reader
            .required("KDA A_log", payload.a_log_addr, heads, heads)
            .await?,
        dt_bias: reader
            .required("KDA dt_bias", payload.dt_bias_addr, key_elements, key_dim)
            .await?,
    };

    let mut state = decode_tensor(
        &buffers.state,
        &buffers.state_scales,
        descriptor.state_precision,
        key_dim,
    )?;
    let mut conv_state = decode_tensor(
        &buffers.conv_state,
        &buffers.conv_state_scales,
        descriptor.conv_state_precision,
        kernel,
    )?;
    let access = VramAccess::new(vram, descriptor.activation_precision)?;
    let mut projection_stats = ProjectionBufferStats::default();

    for token in 0..usize::from(descriptor.valid_tokens) {
        for batch_index in 0..batch {
            let (projected, token_stats) = access
                .read_projection_token(descriptor, token, batch_index, layouts)
                .await?;
            projection_stats.accumulate(token_stats);
            let q_raw =
                &projected[payload.q_offset as usize..payload.q_offset as usize + key_elements];
            let k_raw =
                &projected[payload.k_offset as usize..payload.k_offset as usize + key_elements];
            let v_raw =
                &projected[payload.v_offset as usize..payload.v_offset as usize + value_elements];
            let mut q = vec![0.0; key_elements];
            let mut k = vec![0.0; key_elements];
            let mut v = vec![0.0; value_elements];
            causal_conv(
                q_raw,
                batch_index,
                0,
                channels,
                kernel,
                &mut conv_state,
                &parameters.q_conv_weight,
                parameters.q_conv_bias.as_deref(),
                &mut q,
            );
            causal_conv(
                k_raw,
                batch_index,
                key_elements,
                channels,
                kernel,
                &mut conv_state,
                &parameters.k_conv_weight,
                parameters.k_conv_bias.as_deref(),
                &mut k,
            );
            causal_conv(
                v_raw,
                batch_index,
                2 * key_elements,
                channels,
                kernel,
                &mut conv_state,
                &parameters.v_conv_weight,
                parameters.v_conv_bias.as_deref(),
                &mut v,
            );

            let decay_start = payload.decay_offset as usize;
            let beta_start = payload.beta_offset as usize;
            let mut output = vec![0.0; value_elements];
            for head in 0..heads {
                let q_head = &mut q[head * key_dim..(head + 1) * key_dim];
                let k_head = &mut k[head * key_dim..(head + 1) * key_dim];
                normalize(q_head);
                normalize(k_head);
                let beta = sigmoid(projected[beta_start + head]);
                for key in 0..key_dim {
                    let gate = projected[decay_start + head * key_dim + key];
                    let rate = parameters.a_log[head].exp();
                    let log_decay = payload.gate_lower_bound
                        * sigmoid(rate * (gate + parameters.dt_bias[head * key_dim + key]));
                    let decay = log_decay.exp();
                    for value in 0..value_dim {
                        let index =
                            ((batch_index * heads + head) * value_dim + value) * key_dim + key;
                        state[index] *= decay;
                    }
                }
                for value in 0..value_dim {
                    let row = ((batch_index * heads + head) * value_dim + value) * key_dim;
                    let prediction = (0..key_dim)
                        .map(|key| state[row + key] * k_head[key])
                        .sum::<f32>();
                    let error = beta * (v[head * value_dim + value] - prediction);
                    let mut reduced = 0.0;
                    for key in 0..key_dim {
                        let updated = state[row + key] + error * k_head[key];
                        state[row + key] = updated;
                        reduced += updated * q_head[key];
                    }
                    output[head * value_dim + value] = payload.output_scale * reduced;
                }
            }
            access
                .write_output_token(descriptor, token, batch_index, &output)
                .await?;
        }

        let (restored_state, encoded_state) =
            requantize(&state, descriptor.state_precision, key_dim)?;
        let (restored_conv, encoded_conv) =
            requantize(&conv_state, descriptor.conv_state_precision, kernel)?;
        state = restored_state;
        conv_state = restored_conv;
        buffers.state = encoded_state.values;
        buffers.state_scales = encoded_state.scales;
        buffers.conv_state = encoded_conv.values;
        buffers.conv_state_scales = encoded_conv.scales;
    }
    Ok(projection_stats)
}

#[allow(clippy::too_many_arguments)]
fn causal_conv(
    input: &[f32],
    batch_index: usize,
    channel_offset: usize,
    total_channels: usize,
    kernel: usize,
    state: &mut [f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    for channel in 0..input.len() {
        let base = (batch_index * total_channels + channel_offset + channel) * kernel;
        state.copy_within(base + 1..base + kernel, base);
        state[base + kernel - 1] = input[channel];
        let mut sum = bias.map_or(0.0, |values| values[channel]);
        for index in 0..kernel {
            sum += state[base + index] * weight[channel * kernel + index];
        }
        output[channel] = silu(sum);
    }
}

fn normalize(values: &mut [f32]) {
    let inverse = (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-6)
        .sqrt()
        .recip();
    for value in values {
        *value *= inverse;
    }
}
