//! HBM ↔ SRAM data transfer ("DMA") for MX-format quantized tensors.
//!
//! Models the on-chip DMA / data-movement IP that bridges off-chip HBM
//! and on-chip SRAM. Architecturally this lives on the memory-subsystem
//! side, separate from the compute accelerator — the compute side just
//! issues `H_PREFETCH_*` / `H_STORE_V` instructions and calls into here.
//!
//! Two primitives:
//!
//! - [`transfer_mx_from_hbm`] — HBM → SRAM read (used by `H_PREFETCH_M`
//!   / `H_PREFETCH_V`). Spawns the actual reads on the executor and
//!   returns a [`Receiver`] that yields the resulting tensor.
//! - [`transfer_mx_to_hbm`] — SRAM → HBM writeback (used by
//!   `H_STORE_V`). Runs inline as an async function.
//!
//! Both speak MX (microexponent) directly: when the HBM type is
//! [`MxDataType::Mx`] they fetch / store the element bytes and scale
//! bytes as separate strided streams. The element stream walks HBM at
//! `stride`; the scale stream walks at `stride / element_scale_ratio`
//! so it tracks the corresponding element blocks rather than the
//! elements themselves.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use memory::{ErasedMemoryModel, MemoryModel as _};
use quantize::{MxDataType, QuantTensor};
use runtime::Executor;
use sram::VectorSram;
use tokio::sync::oneshot::{self, Receiver};

/// Process-global quiet flag for debug output emitted from this crate.
///
/// The CLI parses its own `--quiet` flag and forwards it here via
/// [`set_quiet`] at boot so DMA-level diagnostics share the same gate
/// as compute-side diagnostics.
static QUIET: AtomicBool = AtomicBool::new(false);

/// Set the quiet flag for DMA-level debug output.
pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// Query the quiet flag.
pub fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Transfer data from HBM into a SRAM-shaped tensor with a strided
/// loading pattern.
///
/// Spawns the reads on the executor and returns immediately with a
/// [`Receiver`] that yields the assembled [`QuantTensor`] once all
/// reads complete.
///
/// Parameters:
/// - `hbm`: HBM model (will be cloned for the spawned task)
/// - `index`: starting address for element data in HBM
/// - `scale_index`: starting address for scale data in HBM (for MXFP/MXINT types)
/// - `hbm_type`: data type format in HBM
/// - `sram_type`: target data type format for SRAM
/// - `rstride`: stride mode selector (1 = use `stride`, else `load_dim`)
/// - `stride`: stride register value (used when `rstride == 1`)
/// - `load_dim`: number of elements per load
/// - `load_amount`: number of strided loads to perform
/// - `write_amount`: number of loads grouped per SRAM write
pub fn transfer_mx_from_hbm(
    hbm: &Arc<dyn ErasedMemoryModel>,
    index: u64,
    scale_index: u64,
    hbm_type: MxDataType,
    sram_type: MxDataType,
    rstride: u8,
    stride: u32,
    load_dim: u32,
    load_amount: u32,
    write_amount: u32,
) -> Receiver<QuantTensor> {
    // input: load_amount is how many "reads", write_amount is how many sram writes
    // write_dim = load_dim * write_amount per write, repeat for (load_amount / write_amount) times
    assert!(load_dim.is_multiple_of(write_amount));
    assert!(load_amount.is_multiple_of(write_amount));

    let write_dim = load_dim * write_amount; // Number of elements per write to sram
    let num_writes = load_amount / write_amount;
    let (sender, receiver) = oneshot::channel();

    let hbm_clone = hbm.clone();
    let stride = if rstride == 1 { stride } else { load_dim };

    Executor::current().spawn(async move {
        let element_ty = hbm_type.element_type();
        let element_bits = element_ty.size_in_bits();

        // Extract scale bits and block size if Mx type, otherwise use element_bits/1 as default
        let (scale_bits, blocksize) = match hbm_type {
            MxDataType::Mx {
                elem: _,
                scale,
                block,
            } => (scale.size_in_bits(), block),
            _ => (element_bits, 1), // Plain type: each element is "scaled" by 1
        };

        let element_scale_ratio = (element_bits * blocksize as u8) / scale_bits;
        let stride_scale = stride as f32 / element_scale_ratio as f32;
        assert!(element_bits.is_power_of_two());

        let len_in_bits_per_load = element_bits as u32 * load_dim;
        assert!(len_in_bits_per_load.is_multiple_of(8 * 64));
        let len_in_bytes_per_load = len_in_bits_per_load / 8;

        // Calculate scale bytes per load iteration (for Mx types)
        let scale_len_in_bytes_per_load = if let MxDataType::Mx {
            elem: _,
            scale,
            block,
        } = hbm_type
        {
            let scale_bits = scale.size_in_bits();
            assert!(scale_bits.is_power_of_two());
            let scale_len_in_bits_per_load = scale_bits as u32 * (load_dim / block);
            assert!(scale_len_in_bits_per_load.is_multiple_of(8));
            scale_len_in_bits_per_load / 8
        } else {
            0
        };

        // Total bytes for all writes:
        let total_bytes = (len_in_bytes_per_load * write_amount * num_writes) as usize;
        let total_scale_bytes = (scale_len_in_bytes_per_load * write_amount * num_writes) as usize;

        let mut bytes = vec![0u8; total_bytes];
        let mut scale_bytes = vec![0u8; total_scale_bytes];
        let hbm_clone = &hbm_clone;

        enum ChunkType {
            Element {
                offset: usize,
                data: [u8; 64],
                len: usize,
            },
            Scale {
                offset: usize,
                data: [u8; 64],
                len: usize,
            },
        }
        let mut futures =
            FuturesUnordered::<Pin<Box<dyn Future<Output = ChunkType> + Send>>>::new();

        // Outer loop: For each "write". Inner: gather blocks for all loads for this write.
        for write_idx in 0..num_writes {
            for block_idx in 0..write_amount {
                let load_iter = write_idx * write_amount + block_idx;
                let element_addr = index + (load_iter * stride) as u64;
                let scale_addr = scale_index + (load_iter as f32 * stride_scale) as u64;
                let byte_offset = (write_idx * write_amount * len_in_bytes_per_load) as usize
                    + block_idx as usize * len_in_bytes_per_load as usize;
                let scale_byte_offset = (write_idx * write_amount * scale_len_in_bytes_per_load)
                    as usize
                    + block_idx as usize * scale_len_in_bytes_per_load as usize;

                // Element chunks:
                for i in 0..(len_in_bytes_per_load as usize).div_ceil(64) {
                    let chunk_offset = byte_offset + i * 64;
                    let chunk_size = std::cmp::min(64, total_bytes - chunk_offset);
                    let addr = element_addr + (i * 64) as u64;
                    assert!(addr.is_multiple_of(64));
                    futures.push(Box::pin(async move {
                        let data = hbm_clone.read(addr).await;
                        ChunkType::Element {
                            offset: chunk_offset,
                            data,
                            len: chunk_size,
                        }
                    }));
                }

                // Scale chunks (if Mx type)
                if scale_len_in_bytes_per_load > 0 {
                    // Always align to 64-byte chunk boundary for loading
                    // For scale_addr, we fetch the aligned 64-byte block, and mask/select out only what is needed
                    let aligned_scale_addr = (scale_addr / 64) * 64;
                    let within_chunk_offset = (scale_addr % 64) as usize;
                    let chunk_offset = scale_byte_offset; // where to write in scale_bytes
                    futures.push(Box::pin(async move {
                        let data = hbm_clone.read(aligned_scale_addr).await;
                        // Copy out only the relevant bytes for this scale_addr;
                        // scale_len_in_bytes_per_load says how many bytes to copy from within the chunk
                        let end_offset = std::cmp::min(
                            within_chunk_offset + scale_len_in_bytes_per_load as usize,
                            64,
                        );
                        let mut selected = [0u8; 64];
                        let len_to_copy = end_offset - within_chunk_offset;
                        selected[..len_to_copy]
                            .copy_from_slice(&data[within_chunk_offset..end_offset]);
                        ChunkType::Scale {
                            offset: chunk_offset,
                            data: selected,
                            len: len_to_copy,
                        }
                    }));
                }
            }
        }

        // Collect all HBM reads
        while let Some(chunk_result) = futures.next().await {
            match chunk_result {
                ChunkType::Element { offset, data, len } => {
                    bytes[offset..offset + len].copy_from_slice(&data[..len]);
                }
                ChunkType::Scale { offset, data, len } => {
                    scale_bytes[offset..offset + len].copy_from_slice(&data[..len]);
                }
            }
        }

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
        let _ = sender.send(QuantTensor::quantize(full_tensor, sram_type));
    });

    receiver
}

/// Transfer data from VRAM into HBM with a strided writing pattern.
///
/// Parameters:
/// - `hbm`: HBM model (will be cloned for chunk reads/writes)
/// - `vram`: source vector SRAM
/// - `src_addr`: starting address in vector SRAM
/// - `index`: starting address for element data in HBM
/// - `scale_index`: starting address for scale data in HBM (for MXFP/MXINT types)
/// - `hbm_type`: target data type format for HBM
/// - `rstride`: stride mode selector (1 = use `stride`, else `store_dim`)
/// - `stride`: stride register value (used when `rstride == 1`)
/// - `store_dim`: number of elements to store per iteration (VLEN)
/// - `store_amount`: number of strided stores to perform
pub async fn transfer_mx_to_hbm(
    hbm: &Arc<dyn ErasedMemoryModel>,
    vram: &Arc<VectorSram>,
    src_addr: u32,
    index: u64,
    scale_index: u64,
    hbm_type: MxDataType,
    rstride: u8,
    stride: u32,
    store_dim: u32,
    store_amount: u32,
) {
    let hbm_clone = hbm.clone();
    let stride = if rstride == 1 { stride } else { store_dim };

    let element_ty = hbm_type.element_type();
    let element_bits = element_ty.size_in_bits();

    // Extract scale bits and block size if Mx type
    let (scale_bits, blocksize) = match hbm_type {
        MxDataType::Mx {
            elem: _,
            scale,
            block,
        } => (scale.size_in_bits(), block),
        _ => (element_bits, 1),
    };

    let element_scale_ratio = (element_bits * blocksize as u8) / scale_bits;
    let stride_scale = stride as f32 / element_scale_ratio as f32;
    assert!(element_bits.is_power_of_two());

    let len_in_bits_per_store = element_bits as u32 * store_dim;
    assert!(len_in_bits_per_store.is_multiple_of(8 * 64));
    let len_in_bytes_per_store = len_in_bits_per_store / 8;

    // Calculate scale bytes per store iteration (for Mx types)
    let scale_len_in_bytes_per_store = if let MxDataType::Mx {
        elem: _,
        scale,
        block,
    } = hbm_type
    {
        let scale_bits = scale.size_in_bits();
        assert!(scale_bits.is_power_of_two());
        let scale_len_in_bits_per_store = scale_bits as u32 * (store_dim / block);
        assert!(scale_len_in_bits_per_store.is_multiple_of(8));
        scale_len_in_bits_per_store / 8
    } else {
        0
    };

    // Read data from VRAM and convert to HBM format
    for store_iter in 0..store_amount {
        // Read from VRAM
        let src_vram_addr = src_addr + store_iter * store_dim;
        let sram_tensor = vram.read(src_vram_addr).await;

        // Debug: Print VRAM data read
        if !is_quiet() {
            let vram_data = sram_tensor.as_tensor();
            let vram_size = vram_data.size1().unwrap() as usize;
            let vram_slice = unsafe {
                core::slice::from_raw_parts(
                    vram_data.data_ptr() as *const f32,
                    vram_size.min(store_dim as usize),
                )
            };
            eprintln!(
                "[H_STORE_V] Store iter {}: VRAM[{}] -> {} FP32 values",
                store_iter,
                src_vram_addr,
                vram_slice.len()
            );
            eprintln!(
                "  VRAM data (first 8): {:?}",
                &vram_slice[..vram_slice.len().min(8)]
            );
        }

        // Convert from SRAM type to HBM type
        let mut hbm_tensor =
            QuantTensor::quantize(sram_tensor.as_tensor().shallow_clone(), hbm_type);

        // Convert to bytes (element bytes + scale bytes)
        let (element_bytes, scale_bytes) = hbm_tensor.into_bytes();

        // Debug: Print converted HBM data
        if !is_quiet() {
            eprintln!("  Converted to HBM format:");
            eprintln!(
                "    Element bytes: {} bytes (first 16): {:?}",
                element_bytes.len(),
                &element_bytes[..element_bytes.len().min(16)]
            );
            if !scale_bytes.is_empty() {
                eprintln!(
                    "    Scale bytes: {} bytes (expected {}): {:?}",
                    scale_bytes.len(),
                    scale_len_in_bytes_per_store,
                    &scale_bytes[..scale_bytes.len().min(8)]
                );
            }
        }

        // Calculate HBM addresses
        let element_addr = index + (store_iter * stride) as u64;
        let scale_addr = scale_index + (store_iter as f32 * stride_scale) as u64;

        // Write element bytes to HBM (64-byte aligned chunks)
        for i in 0..(len_in_bytes_per_store as usize).div_ceil(64) {
            let chunk_offset = i * 64;
            let chunk_size = std::cmp::min(64, len_in_bytes_per_store as usize - chunk_offset);
            let addr = element_addr + (i * 64) as u64;
            assert!(addr.is_multiple_of(64));

            let mut chunk = [0u8; 64];
            if chunk_offset < element_bytes.len() {
                let copy_len = std::cmp::min(chunk_size, element_bytes.len() - chunk_offset);
                chunk[..copy_len]
                    .copy_from_slice(&element_bytes[chunk_offset..chunk_offset + copy_len]);
            }
            hbm_clone.write(addr, chunk).await;
        }

        // Write scale bytes to HBM (if Mx type)
        // Handle scales that may span multiple 64-byte chunks
        if scale_len_in_bytes_per_store > 0 {
            let mut scale_bytes_written = 0;
            let total_scale_bytes = scale_len_in_bytes_per_store as usize;

            // Write scales in 64-byte chunks, handling unaligned addresses
            while scale_bytes_written < total_scale_bytes {
                let current_scale_addr = scale_addr + scale_bytes_written as u64;
                let aligned_scale_addr = (current_scale_addr / 64) * 64;
                let within_chunk_offset = (current_scale_addr % 64) as usize;

                // Read existing chunk
                let mut existing_chunk = hbm_clone.read(aligned_scale_addr).await;

                // Calculate how many bytes we can write in this chunk
                let bytes_remaining = total_scale_bytes - scale_bytes_written;
                let bytes_in_chunk = std::cmp::min(64 - within_chunk_offset, bytes_remaining);
                let bytes_to_copy =
                    std::cmp::min(bytes_in_chunk, scale_bytes.len() - scale_bytes_written);

                if bytes_to_copy > 0 {
                    // Debug: Print scale write info
                    if !is_quiet() && scale_bytes_written == 0 {
                        eprintln!(
                            "    Writing scale: {} total bytes starting at HBM[0x{:x}]",
                            total_scale_bytes, scale_addr
                        );
                        eprintln!(
                            "      First chunk: {} bytes at HBM[0x{:x}] (offset within chunk: {})",
                            bytes_to_copy, aligned_scale_addr, within_chunk_offset
                        );
                        eprintln!(
                            "      Scale data (hex): {:02x?}",
                            &scale_bytes[scale_bytes_written
                                ..(scale_bytes_written + bytes_to_copy)
                                    .min(scale_bytes_written + 8)]
                        );
                    }

                    // Copy scale bytes into the chunk
                    existing_chunk[within_chunk_offset..within_chunk_offset + bytes_to_copy]
                        .copy_from_slice(
                            &scale_bytes[scale_bytes_written..scale_bytes_written + bytes_to_copy],
                        );

                    // Write the modified chunk back
                    hbm_clone.write(aligned_scale_addr, existing_chunk).await;

                    scale_bytes_written += bytes_to_copy;
                } else {
                    break;
                }
            }

            if !is_quiet() {
                eprintln!(
                    "    Wrote {} scale bytes total (expected {})",
                    scale_bytes_written, total_scale_bytes
                );
                if scale_bytes_written != total_scale_bytes {
                    eprintln!("    ERROR: Scale bytes written mismatch!");
                }
            }
        }

        if !is_quiet() {
            eprintln!("[H_STORE_V] Store iter {} completed\n", store_iter);
        }
    }
}
