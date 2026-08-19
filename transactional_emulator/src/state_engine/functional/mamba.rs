use std::sync::Arc;

use memory::ErasedMemoryModel;
use sram::VectorSram;

use super::{ParameterReader, silu};
use crate::state_engine::cache::StateBuffers;
use crate::state_engine::descriptor::{Mamba2Payload, StateDescriptor};
use crate::state_engine::error::StateEngineError;
use crate::state_engine::layout::LayoutStore;
use crate::state_engine::precision::{decode_tensor, requantize};
use crate::state_engine::projection::ProjectionBufferStats;
use crate::state_engine::vram::VramAccess;

struct MambaParameters {
    conv_weight: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    d_skip: Vec<f32>,
}

pub async fn execute(
    hbm: &Arc<dyn ErasedMemoryModel>,
    vram: &Arc<VectorSram>,
    layouts: &mut LayoutStore,
    descriptor: &StateDescriptor,
    payload: &Mamba2Payload,
    buffers: &mut StateBuffers,
) -> Result<ProjectionBufferStats, StateEngineError> {
    let batch = usize::from(descriptor.batch_size);
    let heads = usize::from(descriptor.num_heads);
    let head_dim = usize::from(payload.head_dim);
    let state_dim = usize::from(payload.state_dim);
    let groups = usize::from(payload.groups);
    let kernel = usize::from(payload.conv_kernel);
    let d_inner = heads * head_dim;
    let group_elements = groups * state_dim;
    let channels = d_inner + 2 * group_elements;

    let mut parameters = ParameterReader::new(
        hbm,
        descriptor.parameter_precision,
        payload.parameter_scale_addr,
    );
    let parameters = MambaParameters {
        conv_weight: parameters
            .required(
                "Mamba conv_weight",
                payload.conv_weight_addr,
                channels * kernel,
                kernel,
            )
            .await?,
        conv_bias: parameters
            .optional(payload.conv_bias_addr, channels, channels)
            .await?,
        a_log: parameters
            .required("Mamba A_log", payload.a_log_addr, heads, heads)
            .await?,
        dt_bias: parameters
            .required("Mamba dt_bias", payload.dt_bias_addr, heads, heads)
            .await?,
        d_skip: parameters
            .required("Mamba D", payload.d_skip_addr, heads, heads)
            .await?,
    };

    let mut state = decode_tensor(
        &buffers.state,
        &buffers.state_scales,
        descriptor.state_precision,
        state_dim,
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
            let xbc_start = payload.xbc_offset as usize;
            let dt_start = payload.dt_offset as usize;
            let mut convolved = vec![0.0; channels];
            for channel in 0..channels {
                let base = (batch_index * channels + channel) * kernel;
                conv_state.copy_within(base + 1..base + kernel, base);
                conv_state[base + kernel - 1] = projected[xbc_start + channel];
                let weight = &parameters.conv_weight[channel * kernel..(channel + 1) * kernel];
                let mut sum = parameters
                    .conv_bias
                    .as_ref()
                    .map_or(0.0, |bias| bias[channel]);
                for index in 0..kernel {
                    sum += conv_state[base + index] * weight[index];
                }
                convolved[channel] = silu(sum);
            }

            let b_group = &convolved[d_inner..d_inner + group_elements];
            let c_group = &convolved[d_inner + group_elements..];
            let mut output = vec![0.0; d_inner];
            for head in 0..heads {
                let dt = softplus(projected[dt_start + head] + parameters.dt_bias[head])
                    .clamp(payload.dt_min, payload.dt_max);
                let decay = (dt * -parameters.a_log[head].exp()).exp();
                let group = head * groups / heads;
                for p in 0..head_dim {
                    let x_index = head * head_dim + p;
                    let x = convolved[x_index];
                    let state_base = ((batch_index * heads + head) * head_dim + p) * state_dim;
                    let mut reduced = 0.0;
                    for n in 0..state_dim {
                        let index = state_base + n;
                        let b = b_group[group * state_dim + n];
                        let c = c_group[group * state_dim + n];
                        let updated = state[index] * decay + dt * b * x;
                        state[index] = updated;
                        reduced += updated * c;
                    }
                    output[x_index] = reduced + x * parameters.d_skip[head];
                }
            }
            access
                .write_output_token(descriptor, token, batch_index, &output)
                .await?;
        }

        let (restored_state, encoded_state) =
            requantize(&state, descriptor.state_precision, state_dim)?;
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

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}
