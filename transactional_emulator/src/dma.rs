//! HBM ↔ SRAM transfer logic for MX-format quantized tensors.
//!
//! This is the MX-aware layer of the accelerator's DMA: it computes the
//! microexponent layout (element vs scale byte streams, strides) and drives
//! the pure byte-movement primitives in [`memory::chunked`] to read from /
//! write to HBM, quantizing along the way.
//!
//! - [`transfer_mx_from_hbm`] — HBM → SRAM read (used by `H_PREFETCH_M` /
//!   `H_PREFETCH_V`). Spawns the reads on the executor and returns a
//!   [`Receiver`] yielding the assembled tensor.
//! - [`transfer_mx_to_hbm`] — SRAM → HBM writeback (used by `H_STORE_V`).
//!   Runs inline as an async function.
//!
//! Both are stateless free functions: HBM and VRAM are passed in by handle,
//! and `stride` is passed in because it lives in the accelerator's register
//! file.

use std::sync::Arc;

use memory::ErasedMemoryModel;
use quantize::{MxDataType, QuantTensor};
use runtime::Executor;
use sram::VectorSram;
use tokio::sync::oneshot::{self, Receiver};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct HbmPhysicalTraffic {
    pub(crate) read_bytes: u64,
    pub(crate) written_bytes: u64,
}

pub(crate) struct MxReadTransfer {
    pub(crate) result: Receiver<QuantTensor>,
    pub(crate) traffic: HbmPhysicalTraffic,
}

/// Derived byte-layout for one MX transfer iteration.
///
/// Computed identically for both transfer directions from the HBM data type,
/// the stride register value, and the per-iteration element count (`load_dim`
/// / `store_dim`).
struct MxLayout {
    element_bits: u8,
    /// Stride for the scale byte stream in physical bytes per iteration.
    scale_stride_bytes: u32,
    /// Element bytes per iteration.
    len_in_bytes: u32,
    /// Scale bytes per iteration (0 for non-MX types).
    scale_len_in_bytes: u32,
}

impl MxLayout {
    fn compute(hbm_type: MxDataType, stride: u32, dim: u32) -> Self {
        let element_bits = hbm_type.element_type().size_in_bits();

        assert!(element_bits.is_power_of_two());
        let element_scale_ratio = hbm_type.element_scale_ratio();
        assert!(stride.is_multiple_of(element_scale_ratio));
        let scale_stride_bytes = stride / element_scale_ratio;

        let len_in_bits = element_bits as u32 * dim;
        // A load must be a whole number of bytes. This was previously required
        // to be a full 64-byte HBM burst (`8 * 64`); relaxed for sub-64 MLEN,
        // which packs fewer than 64 bytes per row.
        assert!(len_in_bits.is_multiple_of(8));
        let len_in_bytes = len_in_bits / 8;

        let scale_len_in_bytes = if let MxDataType::Mx {
            elem: _,
            scale,
            block,
        } = hbm_type
        {
            assert!(dim.is_multiple_of(block));
            let scale_bits = scale.size_in_bits();
            assert!(scale_bits.is_power_of_two());
            let scale_len_in_bits = scale_bits as u32 * (dim / block);
            assert!(scale_len_in_bits.is_multiple_of(8));
            scale_len_in_bits / 8
        } else {
            0
        };

        MxLayout {
            element_bits,
            scale_stride_bytes,
            len_in_bytes,
            scale_len_in_bytes,
        }
    }
}

fn append_chunk_reads(
    reads: &mut Vec<memory::chunked::ChunkRead>,
    source_addr: u64,
    byte_len: usize,
    dst_offset: usize,
) {
    let source_end = source_addr
        .checked_add(byte_len as u64)
        .expect("HBM read range overflow");
    let mut block_addr = source_addr / 64 * 64;
    while block_addr < source_end {
        let copy_start = block_addr.max(source_addr);
        let copy_end = (block_addr + 64).min(source_end);
        reads.push(memory::chunked::ChunkRead {
            addr: copy_start,
            dst_offset: dst_offset + (copy_start - source_addr) as usize,
            len: (copy_end - copy_start) as usize,
        });
        block_addr += 64;
    }
}

fn physical_bytes_for_unaligned_range(address: u64, byte_len: usize) -> u64 {
    if byte_len == 0 {
        return 0;
    }
    let first_block_offset = address % 64;
    (first_block_offset + byte_len as u64).div_ceil(64) * 64
}

/// A strided MX-format region in HBM — the "where + what" of a transfer,
/// independent of the SRAM side.
///
/// Element bytes and scale bytes live in two streams starting at `index` and
/// `scale_index`. All addresses and strides in this interface are physical bytes.
#[derive(Clone, Copy)]
pub(crate) struct MxRegion {
    /// Data type as laid out in HBM.
    pub(crate) hbm_type: MxDataType,
    /// Starting address of the element byte stream.
    pub(crate) index: u64,
    /// Starting address of the scale byte stream (MX types only).
    pub(crate) scale_index: u64,
    /// Stride mode selector: 1 uses `stride`; 0 uses the contiguous byte span.
    pub(crate) rstride: u8,
    /// Stride register value (used when `rstride == 1`).
    pub(crate) stride: u32,
}

/// Transfer data from an HBM [`MxRegion`] into a SRAM-shaped tensor with a
/// strided loading pattern.
///
/// Parameters:
/// - `hbm`: HBM model (cloned into the spawned task)
/// - `region`: the HBM source region (addresses, stride, data type)
/// - `sram_type`: target data type format for SRAM
/// - `load_dim`: number of elements per load
/// - `load_amount`: number of strided loads to perform
/// - `write_amount`: number of loads grouped per SRAM write
pub(crate) fn transfer_mx_from_hbm(
    hbm: &Arc<dyn ErasedMemoryModel>,
    region: MxRegion,
    sram_type: MxDataType,
    load_dim: u32,
    load_amount: u32,
    write_amount: u32,
) -> MxReadTransfer {
    // input: load_amount is how many "reads", write_amount is how many sram writes
    // write_dim = load_dim * write_amount per write, repeat for (load_amount / write_amount) times
    assert!(load_dim.is_multiple_of(write_amount));
    assert!(load_amount.is_multiple_of(write_amount)); // must divide evenly

    let write_dim = load_dim * write_amount; // Number of elements per write to sram
    let num_writes = load_amount / write_amount;
    let (sender, receiver) = oneshot::channel();

    let MxRegion {
        hbm_type,
        index,
        scale_index,
        rstride,
        stride,
    } = region;
    assert!(rstride <= 1, "HBM rstride must be 0 or 1");
    let contiguous_bits = load_dim * hbm_type.element_type().size_in_bits() as u32;
    assert!(contiguous_bits.is_multiple_of(8));
    let stride = if rstride == 1 {
        stride
    } else {
        contiguous_bits / 8
    };
    // Compute issue-origin bytes without moving read-list construction out of
    // the spawned DMA task. Preserving that task boundary keeps asynchronous
    // scheduling identical to the uninstrumented transfer path.
    let issue_layout = MxLayout::compute(hbm_type, stride, load_dim);
    let mut physical_read_bytes = 0_u64;
    for load_iter in 0..load_amount {
        let element_addr = index + (load_iter * stride) as u64;
        let element_bytes =
            physical_bytes_for_unaligned_range(element_addr, issue_layout.len_in_bytes as usize);
        physical_read_bytes = physical_read_bytes
            .checked_add(element_bytes)
            .expect("HBM physical read byte count overflow");
        if issue_layout.scale_len_in_bytes > 0 {
            let scale_addr = scale_index + (load_iter * issue_layout.scale_stride_bytes) as u64;
            let scale_bytes = physical_bytes_for_unaligned_range(
                scale_addr,
                issue_layout.scale_len_in_bytes as usize,
            );
            physical_read_bytes = physical_read_bytes
                .checked_add(scale_bytes)
                .expect("HBM physical read byte count overflow");
        }
    }
    let hbm = hbm.clone();

    Executor::current().spawn(async move {
        let layout = MxLayout::compute(hbm_type, stride, load_dim);
        let element_bits = layout.element_bits;
        let len_in_bytes_per_load = layout.len_in_bytes;
        let scale_len_in_bytes_per_load = layout.scale_len_in_bytes;
        let total_bytes = (len_in_bytes_per_load * write_amount * num_writes) as usize;
        let total_scale_bytes = (scale_len_in_bytes_per_load * write_amount * num_writes) as usize;
        let mut reads = Vec::new();
        for write_idx in 0..num_writes {
            for block_idx in 0..write_amount {
                let load_iter = write_idx * write_amount + block_idx;
                let element_addr = index + (load_iter * stride) as u64;
                let scale_addr = scale_index + (load_iter * layout.scale_stride_bytes) as u64;
                let byte_offset = (write_idx * write_amount * len_in_bytes_per_load) as usize
                    + block_idx as usize * len_in_bytes_per_load as usize;
                let scale_byte_offset = (write_idx * write_amount * scale_len_in_bytes_per_load)
                    as usize
                    + block_idx as usize * scale_len_in_bytes_per_load as usize;
                append_chunk_reads(
                    &mut reads,
                    element_addr,
                    len_in_bytes_per_load as usize,
                    byte_offset,
                );
                if scale_len_in_bytes_per_load > 0 {
                    append_chunk_reads(
                        &mut reads,
                        scale_addr,
                        scale_len_in_bytes_per_load as usize,
                        total_bytes + scale_byte_offset,
                    );
                }
            }
        }
        let gathered = memory::chunked::gather(&hbm, total_bytes + total_scale_bytes, reads).await;
        let bytes = &gathered[..total_bytes];
        let scale_bytes = &gathered[total_bytes..];

        // Process each write batch
        let mut all_results: Vec<QuantTensor> = Vec::with_capacity(num_writes as usize);
        for write_idx in 0..num_writes {
            let write_elements = write_dim as usize;

            let bytes_start = (write_idx * write_amount) as usize * len_in_bytes_per_load as usize;
            let element_bytes = write_elements * element_bits as usize / 8;
            let scale_bytes_start =
                (write_idx * write_amount) as usize * scale_len_in_bytes_per_load as usize;
            let scale_bytes_for_write = match hbm_type {
                MxDataType::Mx { scale, block, .. } => {
                    write_elements / block as usize * scale.size_in_bits() as usize / 8
                }
                MxDataType::Plain(_) => 0,
            };
            let decoded = QuantTensor::from_bytes(
                &bytes[bytes_start..bytes_start + element_bytes],
                &scale_bytes[scale_bytes_start..scale_bytes_start + scale_bytes_for_write],
                write_elements,
                hbm_type,
            );
            all_results.push(QuantTensor::quantize(
                decoded.as_tensor().shallow_clone(),
                sram_type,
            ));
        }

        // Send all results as a concatenated tensor
        // (To maintain compatibility: flatten and send as one QuantTensor)
        let full_tensor = tch::Tensor::cat(
            &all_results
                .iter()
                .map(|qt| qt.as_tensor())
                .collect::<Vec<_>>(),
            0,
        );
        // The receiver may have been dropped if the consumer is no longer
        // interested; that's expected, not worth crashing over — just record it.
        if sender
            .send(QuantTensor::quantize(full_tensor, sram_type))
            .is_err()
        {
            tracing::trace!("HBM->SRAM transfer result discarded: receiver dropped");
        }
    });

    MxReadTransfer {
        result: receiver,
        traffic: HbmPhysicalTraffic {
            read_bytes: physical_read_bytes,
            written_bytes: 0,
        },
    }
}

/// Transfer data from VRAM into an HBM [`MxRegion`] with a strided writing
/// pattern.
///
/// Parameters:
/// - `hbm`: HBM model
/// - `vram`: source vector SRAM
/// - `region`: the HBM destination region (addresses, stride, data type)
/// - `src_addr`: starting address in vector SRAM
/// - `store_dim`: number of elements to store per iteration (VLEN)
/// - `store_amount`: number of strided stores to perform
pub(crate) async fn transfer_mx_to_hbm(
    hbm: &Arc<dyn ErasedMemoryModel>,
    vram: &Arc<VectorSram>,
    region: MxRegion,
    src_addr: u32,
    store_dim: u32,
    store_amount: u32,
) -> HbmPhysicalTraffic {
    let MxRegion {
        hbm_type,
        index,
        scale_index,
        rstride,
        stride,
    } = region;
    assert!(rstride <= 1, "HBM rstride must be 0 or 1");
    let contiguous_bits = store_dim * hbm_type.element_type().size_in_bits() as u32;
    assert!(contiguous_bits.is_multiple_of(8));
    let stride = if rstride == 1 {
        stride
    } else {
        contiguous_bits / 8
    };

    let layout = MxLayout::compute(hbm_type, stride, store_dim);
    let len_in_bytes_per_store = layout.len_in_bytes;
    let scale_len_in_bytes_per_store = layout.scale_len_in_bytes;
    let mut traffic = HbmPhysicalTraffic::default();

    // Read data from VRAM and convert to HBM format
    for store_iter in 0..store_amount {
        // Read from VRAM
        let src_vram_addr = src_addr + store_iter * store_dim;
        let sram_tensor = vram.read(src_vram_addr).await;

        // Debug: Print VRAM data read (trace level — guarded because of unsafe slice)
        if tracing::enabled!(tracing::Level::TRACE) {
            let vram_data = sram_tensor.as_tensor();
            let vram_size = vram_data.size1().unwrap() as usize;
            let vram_slice = unsafe {
                core::slice::from_raw_parts(
                    vram_data.data_ptr() as *const f32,
                    vram_size.min(store_dim as usize),
                )
            };
            tracing::trace!(
                "[H_STORE_V] Store iter {}: VRAM[{}] -> {} FP32 values",
                store_iter,
                src_vram_addr,
                vram_slice.len()
            );
            tracing::trace!(
                "VRAM data (first 8): {:?}",
                &vram_slice[..vram_slice.len().min(8)]
            );
        }

        // Convert from SRAM type to HBM type
        let mut hbm_tensor =
            QuantTensor::quantize(sram_tensor.as_tensor().shallow_clone(), hbm_type);

        // Convert to bytes (element bytes + scale bytes)
        let (element_bytes, scale_bytes) = hbm_tensor.into_bytes();
        assert!(
            element_bytes.len() >= len_in_bytes_per_store as usize,
            "quantized element payload is shorter than the HBM store"
        );
        assert!(
            scale_bytes.len() >= scale_len_in_bytes_per_store as usize,
            "quantized scale payload is shorter than the HBM store"
        );

        // Debug: Print converted HBM data
        tracing::trace!("Converted to HBM format:");
        tracing::trace!(
            "Element bytes: {} bytes (first 16): {:?}",
            element_bytes.len(),
            &element_bytes[..element_bytes.len().min(16)]
        );
        if !scale_bytes.is_empty() {
            tracing::trace!(
                "Scale bytes: {} bytes (expected {}): {:?}",
                scale_bytes.len(),
                scale_len_in_bytes_per_store,
                &scale_bytes[..scale_bytes.len().min(8)]
            );
        }

        // Calculate HBM addresses
        let element_addr = index + (store_iter * stride) as u64;
        let scale_addr = scale_index + (store_iter * layout.scale_stride_bytes) as u64;
        let element_physical_bytes =
            physical_bytes_for_unaligned_range(element_addr, len_in_bytes_per_store as usize);
        traffic.read_bytes = traffic
            .read_bytes
            .checked_add(element_physical_bytes)
            .expect("HBM physical read byte count overflow");
        traffic.written_bytes = traffic
            .written_bytes
            .checked_add(element_physical_bytes)
            .expect("HBM physical write byte count overflow");

        // Write element bytes to HBM via read-modify-write. element_addr need
        // not be 64-aligned (sub-64 MLEN), and write_unaligned avoids
        // clobbering neighbouring bytes. For MLEN >= 64 (element_addr
        // 64-aligned, len a 64-multiple) this is equivalent to write_aligned.
        let _ = memory::chunked::write_unaligned(
            &hbm,
            element_addr,
            len_in_bytes_per_store as usize,
            &element_bytes,
        )
        .await;

        // Write scale bytes to HBM (if Mx type). Handles unaligned addresses
        // and scales that span multiple 64-byte chunks via read-modify-write.
        if scale_len_in_bytes_per_store > 0 {
            let total_scale_bytes = scale_len_in_bytes_per_store as usize;
            let scale_physical_bytes =
                physical_bytes_for_unaligned_range(scale_addr, total_scale_bytes);
            traffic.read_bytes = traffic
                .read_bytes
                .checked_add(scale_physical_bytes)
                .expect("HBM physical read byte count overflow");
            traffic.written_bytes = traffic
                .written_bytes
                .checked_add(scale_physical_bytes)
                .expect("HBM physical write byte count overflow");

            // Debug: describe the first chunk before writing (matches the
            // first-iteration values of the write loop below).
            let within = (scale_addr % 64) as usize;
            let first_chunk = std::cmp::min(
                std::cmp::min(64 - within, total_scale_bytes),
                scale_bytes.len(),
            );
            if first_chunk > 0 {
                tracing::debug!(
                    "Writing scale: {} total bytes starting at HBM[0x{:x}]",
                    total_scale_bytes,
                    scale_addr
                );
                tracing::debug!(
                    "First chunk: {} bytes at HBM[0x{:x}] (offset within chunk: {})",
                    first_chunk,
                    (scale_addr / 64) * 64,
                    within
                );
                tracing::trace!(
                    "Scale data (hex): {:02x?}",
                    &scale_bytes[..first_chunk.min(8)]
                );
            }

            let written =
                memory::chunked::write_unaligned(&hbm, scale_addr, total_scale_bytes, &scale_bytes)
                    .await;

            tracing::debug!(
                "Wrote {} scale bytes total (expected {})",
                written,
                total_scale_bytes
            );
            if written != total_scale_bytes {
                tracing::warn!("Scale bytes written mismatch!");
            }
        }

        tracing::debug!("[H_STORE_V] Store iter {} completed", store_iter);
    }
    traffic
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{DataType, FpType};

    fn e4m3() -> FpType {
        FpType {
            sign: true,
            exponent: 4,
            mantissa: 3,
        }
    }

    #[test]
    fn test_layout_plain_has_no_scale_stream() {
        // Plain(e4m3): 8-bit elements, no scale stream. element_scale_ratio is
        // 1, so the scale stride mirrors the element stride.
        let layout = MxLayout::compute(MxDataType::Plain(DataType::Fp(e4m3())), 64, 64);
        assert_eq!(layout.element_bits, 8);
        assert_eq!(layout.scale_stride_bytes, 64);
        assert_eq!(layout.len_in_bytes, 64); // 8 bits * 64 / 8
        assert_eq!(layout.scale_len_in_bytes, 0);
    }

    #[test]
    fn test_layout_mx_block_scale_stream() {
        // Mx { e4m3 elems, E8M0 scale, block 32 }: ratio = 8*32/8 = 32, so one
        // scale per 32 elements. dim 64 -> 2 scale elements -> 2 bytes.
        let ty = MxDataType::Mx {
            elem: DataType::Fp(e4m3()),
            scale: DataType::Fp(FpType::E8M0),
            block: 32,
        };
        let layout = MxLayout::compute(ty, 64, 64);
        assert_eq!(layout.element_bits, 8);
        assert_eq!(layout.scale_stride_bytes, 2);
        assert_eq!(layout.len_in_bytes, 64); // 8 * 64 / 8
        assert_eq!(layout.scale_len_in_bytes, 2); // 8 bits * (64/32) / 8
    }

    #[test]
    fn test_layout_len_scales_with_dim() {
        // len_in_bytes is element_bits * dim / 8; halving dim halves the length.
        let plain = MxDataType::Plain(DataType::Fp(e4m3()));
        assert_eq!(MxLayout::compute(plain, 32, 32).len_in_bytes, 32);
        assert_eq!(MxLayout::compute(plain, 32, 128).len_in_bytes, 128);
    }

    #[test]
    fn test_layout_16bit_element_doubles_byte_length() {
        // F16 is 16-bit, so len_in_bytes = 16 * dim / 8 = 2 * dim.
        let plain = MxDataType::Plain(DataType::Fp(FpType::F16));
        let layout = MxLayout::compute(plain, 64, 64);
        assert_eq!(layout.element_bits, 16);
        assert_eq!(layout.len_in_bytes, 128); // 16 * 64 / 8
    }

    #[test]
    fn test_layout_subbyte_stride_uses_physical_bytes() {
        for width in [2u32, 4, 8] {
            let ty = MxDataType::Mx {
                elem: DataType::Int(quantize::IntType { width }),
                scale: DataType::Fp(FpType::E8M0),
                block: 8,
            };
            let row_bytes = 2 * width;
            let layout = MxLayout::compute(ty, row_bytes, 16);
            assert_eq!(layout.len_in_bytes, 2 * width);
            assert_eq!(layout.scale_len_in_bytes, 2);
            assert_eq!(layout.scale_stride_bytes, 2);
        }
    }

    #[test]
    fn test_chunk_reads_cover_unaligned_multiblock_range() {
        let mut reads = Vec::new();
        append_chunk_reads(&mut reads, 63, 130, 7);
        assert_eq!(reads.len(), 4);
        assert_eq!(
            (reads[0].addr, reads[0].dst_offset, reads[0].len),
            (63, 7, 1)
        );
        assert_eq!(
            (reads[1].addr, reads[1].dst_offset, reads[1].len),
            (64, 8, 64)
        );
        assert_eq!(
            (reads[2].addr, reads[2].dst_offset, reads[2].len),
            (128, 72, 64)
        );
        assert_eq!(
            (reads[3].addr, reads[3].dst_offset, reads[3].len),
            (192, 136, 1)
        );
        assert_eq!(reads.iter().map(|read| read.len).sum::<usize>(), 130);
    }

    #[test]
    fn test_physical_bytes_cover_every_touched_block() {
        assert_eq!(physical_bytes_for_unaligned_range(0, 0), 0);
        assert_eq!(physical_bytes_for_unaligned_range(0, 64), 64);
        assert_eq!(physical_bytes_for_unaligned_range(63, 1), 64);
        assert_eq!(physical_bytes_for_unaligned_range(63, 2), 128);
        assert_eq!(physical_bytes_for_unaligned_range(65, 128), 192);
    }
}
