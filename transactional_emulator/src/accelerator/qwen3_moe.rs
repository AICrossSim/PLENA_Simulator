//! Exact raw-BF16 HBM layout used by the Qwen3 MoE-tail validation opcode.

use std::sync::Arc;

use half::bf16;
use memory::ErasedMemoryModel;
use memory::chunked::{ChunkRead, gather};
use tch::{Kind, Tensor};

const DESCRIPTOR_BYTES: usize = 64;
const MAGIC: &[u8; 8] = b"Q3MOEBF1";
const VERSION: u32 = 1;
const EXPERTS: usize = 128;
const READ_WINDOW_CHUNKS: usize = 1024;

pub(super) struct ExpertBankDescriptor {
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) gate_up_base: u64,
    pub(super) down_base: u64,
    pub(super) gate_up_stride: usize,
    pub(super) down_stride: usize,
}

pub(super) const fn descriptor_physical_read_bytes() -> u64 {
    DESCRIPTOR_BYTES as u64
}

impl ExpertBankDescriptor {
    pub(super) fn expert_physical_read_bytes(&self) -> u64 {
        u64::try_from(self.gate_up_stride + self.down_stride)
            .expect("Qwen3 expert read byte count does not fit u64")
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

async fn read_exact(hbm: &Arc<dyn ErasedMemoryModel>, base: u64, byte_count: usize) -> Vec<u8> {
    assert!(
        base.is_multiple_of(64),
        "raw BF16 HBM base must be 64-byte aligned"
    );
    assert!(byte_count > 0, "raw BF16 HBM read must be non-empty");
    let mut output = Vec::with_capacity(byte_count);
    let mut consumed = 0usize;
    while consumed < byte_count {
        let window_bytes = (byte_count - consumed).min(READ_WINDOW_CHUNKS * 64);
        let reads = (0..window_bytes.div_ceil(64))
            .map(|chunk| ChunkRead {
                addr: base
                    .checked_add((consumed + chunk * 64) as u64)
                    .expect("raw BF16 HBM read address overflow"),
                dst_offset: chunk * 64,
                len: (window_bytes - chunk * 64).min(64),
            })
            .collect();
        output.extend(gather(hbm, window_bytes, reads).await);
        consumed += window_bytes;
    }
    output
}

pub(super) async fn read_descriptor(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor_base: u64,
) -> ExpertBankDescriptor {
    let bytes = read_exact(hbm, descriptor_base, DESCRIPTOR_BYTES).await;
    assert_eq!(&bytes[..8], MAGIC, "invalid Qwen3 expert descriptor magic");
    assert_eq!(
        read_u32(&bytes, 8),
        VERSION,
        "unsupported Qwen3 expert descriptor version"
    );
    let hidden = read_u32(&bytes, 12) as usize;
    let intermediate = read_u32(&bytes, 16) as usize;
    assert_eq!(
        read_u32(&bytes, 20) as usize,
        EXPERTS,
        "Qwen3 descriptor requires 128 experts"
    );
    assert!(
        matches!((hidden, intermediate), (64, 64) | (2048, 768)),
        "unsupported Qwen3 expert descriptor geometry"
    );
    let gate_up_base = read_u64(&bytes, 24);
    let down_base = read_u64(&bytes, 32);
    let gate_up_stride =
        usize::try_from(read_u64(&bytes, 40)).expect("Qwen3 gate/up stride does not fit usize");
    let down_stride =
        usize::try_from(read_u64(&bytes, 48)).expect("Qwen3 down stride does not fit usize");
    let declared_end = read_u64(&bytes, 56);
    let expected_gate_up_stride = hidden
        .checked_mul(intermediate)
        .and_then(|value| value.checked_mul(4))
        .expect("Qwen3 gate/up stride overflow");
    let expected_down_stride = hidden
        .checked_mul(intermediate)
        .and_then(|value| value.checked_mul(2))
        .expect("Qwen3 down stride overflow");
    assert_eq!(
        gate_up_stride, expected_gate_up_stride,
        "Qwen3 gate/up stride mismatch"
    );
    assert_eq!(
        down_stride, expected_down_stride,
        "Qwen3 down stride mismatch"
    );
    let expected_gate_up_base = descriptor_base
        .checked_add(DESCRIPTOR_BYTES as u64)
        .expect("Qwen3 gate/up base overflow");
    assert_eq!(
        gate_up_base, expected_gate_up_base,
        "Qwen3 descriptor is not canonical"
    );
    let expected_down_base = gate_up_base
        .checked_add((gate_up_stride * EXPERTS) as u64)
        .expect("Qwen3 down base overflow");
    assert_eq!(
        down_base, expected_down_base,
        "Qwen3 descriptor is not canonical"
    );
    let expected_end = down_base
        .checked_add((down_stride * EXPERTS) as u64)
        .expect("Qwen3 expert-bank end overflow");
    assert_eq!(declared_end, expected_end, "Qwen3 descriptor end mismatch");
    assert!(gate_up_base.is_multiple_of(64));
    assert!(down_base.is_multiple_of(64));
    assert!(gate_up_stride.is_multiple_of(64));
    assert!(down_stride.is_multiple_of(64));
    ExpertBankDescriptor {
        hidden,
        intermediate,
        gate_up_base,
        down_base,
        gate_up_stride,
        down_stride,
    }
}

fn bf16_tensor(bytes: &[u8], rows: usize, cols: usize, label: &str) -> Tensor {
    let elements = rows
        .checked_mul(cols)
        .expect("raw BF16 tensor size overflow");
    assert_eq!(
        bytes.len(),
        elements * 2,
        "{label} raw BF16 byte count mismatch"
    );
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite BF16 value"
    );
    Tensor::from_slice(&values)
        .reshape([rows as i64, cols as i64])
        .to_kind(Kind::BFloat16)
}

pub(super) async fn read_expert(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &ExpertBankDescriptor,
    expert_id: u32,
) -> (Tensor, Tensor) {
    let expert = usize::try_from(expert_id).unwrap();
    assert!(expert < EXPERTS, "Qwen3 expert ID is out of range");
    let gate_up_addr = descriptor
        .gate_up_base
        .checked_add((expert * descriptor.gate_up_stride) as u64)
        .expect("Qwen3 gate/up expert address overflow");
    let down_addr = descriptor
        .down_base
        .checked_add((expert * descriptor.down_stride) as u64)
        .expect("Qwen3 down expert address overflow");
    let (gate_up_bytes, down_bytes) = tokio::join!(
        read_exact(hbm, gate_up_addr, descriptor.gate_up_stride),
        read_exact(hbm, down_addr, descriptor.down_stride),
    );
    (
        bf16_tensor(
            &gate_up_bytes,
            2 * descriptor.intermediate,
            descriptor.hidden,
            "Qwen3 fused gate/up bank",
        ),
        bf16_tensor(
            &down_bytes,
            descriptor.hidden,
            descriptor.intermediate,
            "Qwen3 down bank",
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory::MemoryBacked;

    #[tokio::test]
    #[should_panic(expected = "invalid Qwen3 expert descriptor magic")]
    async fn malformed_descriptor_magic_fails_closed() {
        let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(MemoryBacked::with_capacity(64));
        let _ = read_descriptor(&hbm, 0).await;
    }

    #[test]
    #[should_panic(expected = "contains a non-finite BF16 value")]
    fn nonfinite_raw_bf16_weight_fails_closed() {
        let bytes = bf16::NAN.to_bits().to_le_bytes();
        let _ = bf16_tensor(&bytes, 1, 1, "fixture");
    }

    #[test]
    fn tiny_issue_origin_bytes_match_aligned_raw_bf16_reads() {
        let descriptor = ExpertBankDescriptor {
            hidden: 64,
            intermediate: 64,
            gate_up_base: 64,
            down_base: 2_097_216,
            gate_up_stride: 16_384,
            down_stride: 8_192,
        };
        assert_eq!(descriptor_physical_read_bytes(), 64);
        assert_eq!(descriptor.expert_physical_read_bytes(), 24_576);
        assert_eq!(
            descriptor_physical_read_bytes() + 8 * descriptor.expert_physical_read_bytes(),
            196_672
        );
    }
}
