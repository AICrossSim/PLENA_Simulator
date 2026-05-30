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
use quantize::{DataType, MxDataType, QuantTensor};
use runtime::Executor;
use sram::VectorSram;
use tokio::sync::oneshot::{self, Receiver};

/// Derived byte-layout for one MX transfer iteration.
///
/// Computed identically for both transfer directions from the HBM data type,
/// the stride register value, and the per-iteration element count (`load_dim`
/// / `store_dim`).
struct MxLayout {
    element_ty: DataType,
    element_bits: u8,
    /// Scale element bit-width (equals `element_bits` for non-MX types, where
    /// it is unused).
    scale_bits: u8,
    /// Stride for the scale byte stream, in scale-elements per iteration.
    stride_scale: f32,
    /// Element bytes per iteration.
    len_in_bytes: u32,
    /// Scale bytes per iteration (0 for non-MX types).
    scale_len_in_bytes: u32,
}

impl MxLayout {
    fn compute(hbm_type: MxDataType, stride: u32, dim: u32) -> Self {
        let element_ty = hbm_type.element_type();
        let element_bits = element_ty.size_in_bits();

        // Scale element bit-width (element_bits for plain types, where the
        // scale stream is unused).
        let scale_bits = match hbm_type {
            MxDataType::Mx { scale, .. } => scale.size_in_bits(),
            _ => element_bits,
        };

        let stride_scale = stride as f32 / hbm_type.element_scale_ratio() as f32;
        assert!(element_bits.is_power_of_two());

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
            let scale_bits = scale.size_in_bits();
            assert!(scale_bits.is_power_of_two());
            let scale_len_in_bits = scale_bits as u32 * (dim / block);
            assert!(scale_len_in_bits.is_multiple_of(8));
            scale_len_in_bits / 8
        } else {
            0
        };

        MxLayout {
            element_ty,
            element_bits,
            scale_bits,
            stride_scale,
            len_in_bytes,
            scale_len_in_bytes,
        }
    }
}

/// A strided MX-format region in HBM — the "where + what" of a transfer,
/// independent of the SRAM side.
///
/// Element bytes and scale bytes (for MX types) live in two streams starting
/// at `index` / `scale_index`; consecutive transfer iterations advance by
/// `stride` (when `rstride == 1`) or by the per-iteration element count.
#[derive(Clone, Copy)]
pub(crate) struct MxRegion {
    /// Data type as laid out in HBM.
    pub(crate) hbm_type: MxDataType,
    /// Starting address of the element byte stream.
    pub(crate) index: u64,
    /// Starting address of the scale byte stream (MX types only).
    pub(crate) scale_index: u64,
    /// Stride mode selector: 1 = use `stride`, else the per-iteration dim.
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
) -> Receiver<QuantTensor> {
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
    let stride = if rstride == 1 { stride } else { load_dim };
    let hbm = hbm.clone();

    Executor::current().spawn(async move {
        let layout = MxLayout::compute(hbm_type, stride, load_dim);
        let element_ty = layout.element_ty;
        let element_bits = layout.element_bits;
        let scale_bits = layout.scale_bits;
        let len_in_bytes_per_load = layout.len_in_bytes;
        let scale_len_in_bytes_per_load = layout.scale_len_in_bytes;

        // Total bytes for all writes:
        let total_bytes = (len_in_bytes_per_load * write_amount * num_writes) as usize;
        let total_scale_bytes = (scale_len_in_bytes_per_load * write_amount * num_writes) as usize;

        // Build the read list. Element + scale reads share one gather pool so
        // they race exactly as a single batch. Scale bytes land in the gather
        // buffer after the element region; the two are split out afterward.
        let mut reads = Vec::new();
        for write_idx in 0..num_writes {
            for block_idx in 0..write_amount {
                let load_iter = write_idx * write_amount + block_idx;
                let element_addr = index + (load_iter * stride) as u64;
                let scale_addr = scale_index + (load_iter as f32 * layout.stride_scale) as u64;
                let byte_offset = (write_idx * write_amount * len_in_bytes_per_load) as usize
                    + block_idx as usize * len_in_bytes_per_load as usize;
                let scale_byte_offset = (write_idx * write_amount * scale_len_in_bytes_per_load)
                    as usize
                    + block_idx as usize * scale_len_in_bytes_per_load as usize;

                // Element chunks: walk the byte range
                // [element_addr, element_addr + len_in_bytes_per_load) one
                // 64-byte block at a time, emitting a ChunkRead clamped to each
                // block's boundaries. `gather` truncates a read at the block
                // end, so no single ChunkRead may straddle a boundary. For
                // MLEN >= 64 (element_addr 64-aligned, len a 64-multiple) this
                // reduces to exactly one full-64-byte read per block.
                let element_end = element_addr + len_in_bytes_per_load as u64;
                let mut blk = (element_addr / 64) * 64;
                while blk < element_end {
                    let copy_start = std::cmp::max(blk, element_addr);
                    let copy_end = std::cmp::min(blk + 64, element_end);
                    let addr = copy_start;
                    let dst_offset = byte_offset + (copy_start - element_addr) as usize;
                    // Clamp against the gather buffer end (matches the previous
                    // `min(64, total_bytes - chunk_offset)` behaviour).
                    let mut len = (copy_end - copy_start) as usize;
                    if dst_offset + len > total_bytes {
                        len = total_bytes - dst_offset;
                    }
                    reads.push(memory::chunked::ChunkRead {
                        addr,
                        dst_offset,
                        len,
                    });
                    blk += 64;
                }

                // Scale chunk (if Mx type). The byte primitive fetches the
                // aligned 64-byte block and slices [within .. within + len].
                if scale_len_in_bytes_per_load > 0 {
                    let within = (scale_addr % 64) as usize;
                    let end = std::cmp::min(within + scale_len_in_bytes_per_load as usize, 64);
                    reads.push(memory::chunked::ChunkRead {
                        addr: scale_addr,
                        dst_offset: total_bytes + scale_byte_offset,
                        len: end - within,
                    });
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

            let mut vec = vec![0f32; write_elements];

            // Fill `vec` with elements for this write
            let bytes_start = (write_idx * write_amount) as usize * len_in_bytes_per_load as usize;

            element_ty.convert_bytes_to_f32_vec(
                &bytes[bytes_start..bytes_start + write_elements * (element_bits as usize / 8)],
                &mut vec,
            );

            // Apply scaling if needed
            if let MxDataType::Mx {
                elem: _,
                scale,
                block,
            } = hbm_type
            {
                let nblocks = write_elements / block as usize;
                let scale_bytes_start =
                    (write_idx * write_amount) as usize * scale_len_in_bytes_per_load as usize;
                let mut scale_vec = vec![0f32; nblocks];
                scale.convert_bytes_to_f32_vec(
                    &scale_bytes[scale_bytes_start
                        ..scale_bytes_start + nblocks * (scale_bits as usize / 8)],
                    &mut scale_vec,
                );
                for (elem_block, scale_val) in vec
                    .chunks_mut(block as usize)
                    .zip(scale_vec.iter().copied())
                {
                    for elem in elem_block.iter_mut() {
                        *elem *= scale_val;
                    }
                }
            }

            let tensor = tch::Tensor::from_slice(&vec);
            all_results.push(QuantTensor::quantize(tensor, sram_type));
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

    receiver
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
) {
    let MxRegion {
        hbm_type,
        index,
        scale_index,
        rstride,
        stride,
    } = region;
    let stride = if rstride == 1 { stride } else { store_dim };

    let layout = MxLayout::compute(hbm_type, stride, store_dim);
    let len_in_bytes_per_store = layout.len_in_bytes;
    let scale_len_in_bytes_per_store = layout.scale_len_in_bytes;

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
        let scale_addr = scale_index + (store_iter as f32 * layout.stride_scale) as u64;

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
}
