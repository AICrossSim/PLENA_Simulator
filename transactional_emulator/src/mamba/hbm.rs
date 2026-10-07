//! HBM serialization for the pure functional Mamba kernel.

use std::sync::Arc;

use half::bf16;
use memory::ErasedMemoryModel;

use crate::generated_contract as contract;

use super::functional::{FunctionalMambaState, FunctionalMambaWeights, execute as execute_kernel};
use super::{MambaDescriptor, MambaError, PrecisionPolicy};

fn checked_usize(value: u64) -> Result<usize, MambaError> {
    value
        .try_into()
        .map_err(|_| MambaError::UnsupportedProfile("HBM span exceeds host address space"))
}

fn checked_bytes(elements: u64, element_bytes: u64) -> Result<usize, MambaError> {
    checked_usize(
        elements
            .checked_mul(element_bytes)
            .ok_or(MambaError::AddressError("HBM span size overflows u64"))?,
    )
}

pub(super) async fn read_bytes(
    hbm: &Arc<dyn ErasedMemoryModel>,
    address: u64,
    size_bytes: usize,
) -> Result<Vec<u8>, MambaError> {
    let mut output = vec![0u8; size_bytes];
    let mut offset = 0usize;
    while offset < size_bytes {
        let current = address
            .checked_add(offset as u64)
            .ok_or(MambaError::AddressError("HBM read address overflows u64"))?;
        let within = current as usize % 64;
        let length = (64 - within).min(size_bytes - offset);
        let block = hbm.box_functional_read(current - within as u64).await;
        output[offset..offset + length].copy_from_slice(&block[within..within + length]);
        offset += length;
    }
    Ok(output)
}

pub(super) async fn write_bytes(
    hbm: &Arc<dyn ErasedMemoryModel>,
    address: u64,
    bytes: &[u8],
) -> Result<(), MambaError> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let current = address
            .checked_add(offset as u64)
            .ok_or(MambaError::AddressError("HBM write address overflows u64"))?;
        let within = current as usize % 64;
        let length = (64 - within).min(bytes.len() - offset);
        let aligned = current - within as u64;
        let mut block = if within == 0 && length == 64 {
            [0u8; 64]
        } else {
            hbm.box_functional_read(aligned).await
        };
        block[within..within + length].copy_from_slice(&bytes[offset..offset + length]);
        hbm.box_functional_write(aligned, block).await;
        offset += length;
    }
    Ok(())
}

fn decode_storage(bytes: &[u8], policy: PrecisionPolicy) -> Result<Vec<f32>, MambaError> {
    let element_bytes = policy.element_bytes() as usize;
    if !bytes.len().is_multiple_of(element_bytes) {
        return Err(MambaError::Internal(
            "storage byte count is not a whole number of elements",
        ));
    }
    Ok(match policy {
        PrecisionPolicy::Fp32Reference => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        PrecisionPolicy::Bf16ActivationFp32State => bytes
            .chunks_exact(2)
            .map(|chunk| {
                f32::from(bf16::from_bits(u16::from_le_bytes(
                    chunk.try_into().unwrap(),
                )))
            })
            .collect(),
    })
}

fn encode_storage(values: &[f32], policy: PrecisionPolicy) -> Vec<u8> {
    match policy {
        PrecisionPolicy::Fp32Reference => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        PrecisionPolicy::Bf16ActivationFp32State => values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect(),
    }
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, MambaError> {
    decode_storage(bytes, PrecisionPolicy::Fp32Reference)
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    encode_storage(values, PrecisionPolicy::Fp32Reference)
}

pub(super) async fn write_completion(
    hbm: &Arc<dyn ErasedMemoryModel>,
    address: u64,
    status: u32,
    completion_event: u32,
    elapsed_cycles: u64,
) -> Result<(), MambaError> {
    let mut bytes = [0u8; contract::MAMBA_COMPLETION_BYTES];
    bytes[contract::MAMBA_CPL_STATUS_OFFSET..contract::MAMBA_CPL_STATUS_OFFSET + 4]
        .copy_from_slice(&status.to_le_bytes());
    bytes[contract::MAMBA_CPL_COMPLETION_EVENT_OFFSET
        ..contract::MAMBA_CPL_COMPLETION_EVENT_OFFSET + 4]
        .copy_from_slice(&completion_event.to_le_bytes());
    bytes[contract::MAMBA_CPL_ELAPSED_CYCLES_OFFSET..contract::MAMBA_CPL_ELAPSED_CYCLES_OFFSET + 8]
        .copy_from_slice(&elapsed_cycles.to_le_bytes());
    write_bytes(hbm, address, &bytes).await
}

async fn load_contiguous(
    hbm: &Arc<dyn ErasedMemoryModel>,
    address: u64,
    elements: u64,
    policy: PrecisionPolicy,
) -> Result<Vec<f32>, MambaError> {
    let bytes = read_bytes(
        hbm,
        address,
        checked_bytes(elements, policy.element_bytes())?,
    )
    .await?;
    decode_storage(&bytes, policy)
}

async fn load_optional(
    hbm: &Arc<dyn ErasedMemoryModel>,
    address: u64,
    elements: u64,
    policy: PrecisionPolicy,
) -> Result<Option<Vec<f32>>, MambaError> {
    if address == 0 {
        Ok(None)
    } else {
        Ok(Some(load_contiguous(hbm, address, elements, policy).await?))
    }
}

async fn load_input(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &MambaDescriptor,
) -> Result<Vec<f32>, MambaError> {
    let row_bytes = checked_bytes(
        descriptor.d_model as u64,
        descriptor.precision.element_bytes(),
    )?;
    let row_elements = descriptor.d_model as usize;
    let mut input =
        vec![
            0.0f32;
            descriptor.batch_size as usize * descriptor.sequence_length as usize * row_elements
        ];
    for batch in 0..descriptor.batch_size as usize {
        for token in 0..descriptor.sequence_length as usize {
            let address = descriptor
                .input_addr
                .checked_add(batch as u64 * descriptor.input_batch_stride as u64)
                .and_then(|value| {
                    value.checked_add(token as u64 * descriptor.input_token_stride as u64)
                })
                .ok_or(MambaError::AddressError("input row address overflows u64"))?;
            let row = decode_storage(
                &read_bytes(hbm, address, row_bytes).await?,
                descriptor.precision,
            )?;
            let start = (batch * descriptor.sequence_length as usize + token) * row_elements;
            input[start..start + row_elements].copy_from_slice(&row);
        }
    }
    Ok(input)
}

async fn load_weights(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &MambaDescriptor,
) -> Result<FunctionalMambaWeights, MambaError> {
    let projection = descriptor.projection_size()?;
    let conv_channels = descriptor.conv_channels()?;
    let output_projection = descriptor.output_projection_size()?;
    let policy = descriptor.precision;
    Ok(FunctionalMambaWeights {
        in_proj_weight: load_contiguous(
            hbm,
            descriptor.in_proj_weight_addr,
            projection * descriptor.d_model as u64,
            policy,
        )
        .await?,
        in_proj_bias: load_optional(hbm, descriptor.in_proj_bias_addr, projection, policy).await?,
        conv_weight: load_contiguous(
            hbm,
            descriptor.conv_weight_addr,
            conv_channels * descriptor.conv_kernel as u64,
            policy,
        )
        .await?,
        conv_bias: load_optional(hbm, descriptor.conv_bias_addr, conv_channels, policy).await?,
        a_log: load_contiguous(
            hbm,
            descriptor.a_log_addr,
            descriptor.num_heads as u64,
            policy,
        )
        .await?,
        dt_bias: load_contiguous(
            hbm,
            descriptor.dt_bias_addr,
            descriptor.num_heads as u64,
            policy,
        )
        .await?,
        d_skip: load_contiguous(
            hbm,
            descriptor.d_skip_addr,
            descriptor.num_heads as u64,
            policy,
        )
        .await?,
        norm_weight: load_contiguous(
            hbm,
            descriptor.norm_weight_addr,
            descriptor.d_inner as u64,
            policy,
        )
        .await?,
        out_proj_weight: load_contiguous(
            hbm,
            descriptor.out_proj_weight_addr,
            descriptor.d_model as u64 * output_projection,
            policy,
        )
        .await?,
        out_proj_bias: load_optional(
            hbm,
            descriptor.out_proj_bias_addr,
            descriptor.d_model as u64,
            policy,
        )
        .await?,
    })
}

async fn load_state(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &MambaDescriptor,
) -> Result<FunctionalMambaState, MambaError> {
    let ssm_bytes = checked_bytes(
        descriptor.batch_size as u64 * descriptor.state_request_stride as u64,
        1,
    )?;
    let conv_bytes = checked_bytes(
        descriptor.batch_size as u64 * descriptor.conv_request_stride as u64,
        1,
    )?;
    Ok(FunctionalMambaState {
        ssm: decode_f32(&read_bytes(hbm, descriptor.ssm_state_addr, ssm_bytes).await?)?,
        conv: decode_f32(&read_bytes(hbm, descriptor.conv_state_addr, conv_bytes).await?)?,
    })
}

async fn write_output(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &MambaDescriptor,
    output: &[f32],
) -> Result<(), MambaError> {
    let row_elements = descriptor.d_model as usize;
    for batch in 0..descriptor.batch_size as usize {
        for token in 0..descriptor.sequence_length as usize {
            let address = descriptor
                .output_addr
                .checked_add(batch as u64 * descriptor.output_batch_stride as u64)
                .and_then(|value| {
                    value.checked_add(token as u64 * descriptor.output_token_stride as u64)
                })
                .ok_or(MambaError::AddressError("output row address overflows u64"))?;
            let start = (batch * descriptor.sequence_length as usize + token) * row_elements;
            let row = encode_storage(&output[start..start + row_elements], descriptor.precision);
            write_bytes(hbm, address, &row).await?;
        }
    }
    Ok(())
}

async fn write_state(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &MambaDescriptor,
    state: &FunctionalMambaState,
) -> Result<(), MambaError> {
    let ssm = encode_f32(&state.ssm);
    let conv = encode_f32(&state.conv);
    write_bytes(hbm, descriptor.ssm_state_addr, &ssm).await?;
    write_bytes(hbm, descriptor.conv_state_addr, &conv).await?;
    Ok(())
}

/// Execute one already-validated descriptor against HBM.
///
/// The caller must run both descriptor and memory-map validation first. This
/// function performs every fallible read and all numerical work before the
/// first architected output/state write.
pub(crate) async fn execute_functional_hbm(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &MambaDescriptor,
    subop: contract::MambaSubop,
) -> Result<(), MambaError> {
    match subop {
        contract::MambaSubop::Wait => Ok(()),
        contract::MambaSubop::StateReset => {
            let state = FunctionalMambaState::zeros(descriptor)?;
            write_state(hbm, descriptor, &state).await
        }
        contract::MambaSubop::Prefill | contract::MambaSubop::Step => {
            let input = load_input(hbm, descriptor).await?;
            let weights = load_weights(hbm, descriptor).await?;
            let initial_state = if descriptor.continue_state() {
                Some(load_state(hbm, descriptor).await?)
            } else {
                None
            };
            let result = execute_kernel(descriptor, &input, &weights, initial_state.as_ref())?;
            write_output(hbm, descriptor, &result.output).await?;
            write_state(hbm, descriptor, &result.state).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory::MemoryBacked;

    fn descriptor() -> MambaDescriptor {
        MambaDescriptor {
            flags: 0,
            context_id: 1,
            sequence_length: 3,
            batch_size: 1,
            d_model: 4,
            d_inner: 4,
            num_heads: 2,
            head_dim: 2,
            state_dim: 2,
            groups: 1,
            chunk_size: 2,
            conv_kernel: 2,
            precision: PrecisionPolicy::Fp32Reference,
            rms_norm_eps: 1e-5,
            input_addr: 0x1000,
            output_addr: 0x1100,
            in_proj_weight_addr: 0x2000,
            in_proj_bias_addr: 0x2100,
            conv_weight_addr: 0x2200,
            conv_bias_addr: 0x2300,
            a_log_addr: 0x2400,
            d_skip_addr: 0x2500,
            norm_weight_addr: 0x2600,
            out_proj_weight_addr: 0x2700,
            out_proj_bias_addr: 0x2800,
            ssm_state_addr: 0x2900,
            conv_state_addr: 0x2a00,
            scratch_addr: 0,
            completion_addr: 0,
            dt_bias_addr: 0x2c00,
            input_batch_stride: 48,
            input_token_stride: 16,
            output_batch_stride: 48,
            output_token_stride: 16,
            state_request_stride: 32,
            state_head_stride: 16,
            conv_request_stride: 64,
            scratch_bytes: 0,
            dependency_event: contract::MAMBA_NO_EVENT,
            completion_event: contract::MAMBA_NO_EVENT,
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

    fn put(data: &mut [u8], address: u64, values: &[f32]) {
        let bytes = encode_f32(values);
        let start = address as usize;
        data[start..start + bytes.len()].copy_from_slice(&bytes);
    }

    #[test]
    fn storage_codecs_preserve_fp32_and_bf16_contracts() {
        let values = [1.0, -0.33333334, 0.0, f32::INFINITY];
        assert_eq!(
            decode_storage(
                &encode_storage(&values, PrecisionPolicy::Fp32Reference),
                PrecisionPolicy::Fp32Reference
            )
            .unwrap(),
            values
        );
        let decoded = decode_storage(
            &encode_storage(&values, PrecisionPolicy::Bf16ActivationFp32State),
            PrecisionPolicy::Bf16ActivationFp32State,
        )
        .unwrap();
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], f32::from(bf16::from_f32(values[1])));
        assert_eq!(decoded[2], 0.0);
        assert!(decoded[3].is_infinite());
    }

    #[tokio::test]
    async fn byte_io_handles_unaligned_cross_block_ranges() {
        let memory = Arc::new(MemoryBacked::with_capacity(192));
        memory.with_data(|bytes| bytes.fill(0xaa));
        let hbm: Arc<dyn ErasedMemoryModel> = memory.clone();
        let payload = (0u8..100).collect::<Vec<_>>();
        write_bytes(&hbm, 31, &payload).await.unwrap();
        assert_eq!(read_bytes(&hbm, 31, payload.len()).await.unwrap(), payload);
        memory.with_data(|bytes| {
            assert_eq!(bytes[30], 0xaa);
            assert_eq!(bytes[131], 0xaa);
        });
    }

    #[tokio::test]
    async fn completion_codec_uses_shared_little_endian_offsets() {
        let memory = Arc::new(MemoryBacked::with_capacity(128));
        let hbm: Arc<dyn ErasedMemoryModel> = memory;
        write_completion(&hbm, 16, 4, 9, 1234).await.unwrap();
        let bytes = read_bytes(&hbm, 16, contract::MAMBA_COMPLETION_BYTES)
            .await
            .unwrap();
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 9);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1234);
    }

    #[tokio::test]
    async fn functional_hbm_prefill_writes_output_state_and_reset() {
        let descriptor = descriptor();
        descriptor
            .validate(contract::MambaSubop::Prefill, 1)
            .unwrap();
        descriptor
            .validate_memory_map(contract::MambaSubop::Prefill, 0x2d00, Some(0x4000))
            .unwrap();
        let input = values(3 * 4, 0.05, 1);
        let weights = weights();
        let expected = execute_kernel(&descriptor, &input, &weights, None).unwrap();

        let memory = Arc::new(MemoryBacked::with_capacity(0x4000));
        memory.with_data(|data| {
            put(data, descriptor.input_addr, &input);
            put(
                data,
                descriptor.in_proj_weight_addr,
                &weights.in_proj_weight,
            );
            put(
                data,
                descriptor.in_proj_bias_addr,
                weights.in_proj_bias.as_deref().unwrap(),
            );
            put(data, descriptor.conv_weight_addr, &weights.conv_weight);
            put(
                data,
                descriptor.conv_bias_addr,
                weights.conv_bias.as_deref().unwrap(),
            );
            put(data, descriptor.a_log_addr, &weights.a_log);
            put(data, descriptor.dt_bias_addr, &weights.dt_bias);
            put(data, descriptor.d_skip_addr, &weights.d_skip);
            put(data, descriptor.norm_weight_addr, &weights.norm_weight);
            put(
                data,
                descriptor.out_proj_weight_addr,
                &weights.out_proj_weight,
            );
            put(
                data,
                descriptor.out_proj_bias_addr,
                weights.out_proj_bias.as_deref().unwrap(),
            );
        });
        let hbm: Arc<dyn ErasedMemoryModel> = memory.clone();
        execute_functional_hbm(&hbm, &descriptor, contract::MambaSubop::Prefill)
            .await
            .unwrap();

        let output =
            decode_f32(&read_bytes(&hbm, descriptor.output_addr, 48).await.unwrap()).unwrap();
        let ssm = decode_f32(
            &read_bytes(&hbm, descriptor.ssm_state_addr, 32)
                .await
                .unwrap(),
        )
        .unwrap();
        let conv = decode_f32(
            &read_bytes(&hbm, descriptor.conv_state_addr, 64)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(output, expected.output);
        assert_eq!(ssm, expected.state.ssm);
        assert_eq!(conv, expected.state.conv);

        execute_functional_hbm(&hbm, &descriptor, contract::MambaSubop::StateReset)
            .await
            .unwrap();
        assert!(
            decode_f32(
                &read_bytes(&hbm, descriptor.ssm_state_addr, 32)
                    .await
                    .unwrap()
            )
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
        );
        assert!(
            decode_f32(
                &read_bytes(&hbm, descriptor.conv_state_addr, 64)
                    .await
                    .unwrap()
            )
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
        );
    }
}
