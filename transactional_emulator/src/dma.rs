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
use std::sync::atomic::{AtomicU64, Ordering};

use memory::ErasedMemoryModel;
use quantize::{DataType, MxDataType, QuantTensor};
use runtime::Executor;
use sram::VectorSram;
use tokio::sync::oneshot::{self, Receiver};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct DmaStatisticsSnapshot {
    pub(crate) read_operations: u64,
    pub(crate) write_operations: u64,
    pub(crate) read_logical_elements: u64,
    pub(crate) write_logical_elements: u64,
    pub(crate) read_useful_payload_bytes: u64,
    pub(crate) write_useful_payload_bytes: u64,
    pub(crate) read_packed_transfer_bytes: u64,
    pub(crate) write_packed_transfer_bytes: u64,
    /// Coalesced 64-byte requests presented to the memory model. Write-side
    /// read-modify-write traffic is intentionally reflected in these counts.
    pub(crate) read_coalesced_line_requests: u64,
    pub(crate) write_coalesced_line_requests: u64,
    pub(crate) read_coalesced_line_bytes: u64,
    pub(crate) write_coalesced_line_bytes: u64,
    /// Native DRAM transfers below the 64-byte memory-model interface. For the
    /// HBM2 preset a 64-byte line is split into four 16-byte physical bursts.
    pub(crate) read_physical_burst_bytes: u64,
    pub(crate) write_physical_burst_bytes: u64,
    pub(crate) read_physical_bursts: u64,
    pub(crate) write_physical_bursts: u64,
}

#[derive(Debug, Default)]
pub(crate) struct DmaStatistics {
    read_operations: AtomicU64,
    write_operations: AtomicU64,
    read_logical_elements: AtomicU64,
    write_logical_elements: AtomicU64,
    read_useful_payload_bytes: AtomicU64,
    write_useful_payload_bytes: AtomicU64,
    read_packed_transfer_bytes: AtomicU64,
    write_packed_transfer_bytes: AtomicU64,
}

impl DmaStatistics {
    fn transfer_sizes(hbm_type: MxDataType, dim: u32, amount: u32) -> (u64, u64, u64) {
        let logical_elements = u64::from(dim) * u64::from(amount);
        let element_bits = u64::from(hbm_type.element_type().size_in_bits());
        let (scale_elements, scale_bits) = match hbm_type {
            MxDataType::Mx { scale, block, .. } => {
                assert!(dim.is_multiple_of(block));
                (
                    u64::from(dim / block) * u64::from(amount),
                    u64::from(scale.size_in_bits()),
                )
            }
            MxDataType::Plain(_) => (0, 0),
        };
        let useful_bits = logical_elements * element_bits + scale_elements * scale_bits;
        let useful_payload_bytes = useful_bits.div_ceil(8);

        // Row-wise packing is what the DMA actually transfers. It can include
        // more padding than a hypothetical tensor-wide bit stream.
        let packed_per_row = u64::from(packed_bytes(dim, element_bits as u8))
            + match hbm_type {
                MxDataType::Mx { scale, block, .. } => {
                    u64::from(packed_bytes(dim / block, scale.size_in_bits()))
                }
                MxDataType::Plain(_) => 0,
            };
        let packed_transfer_bytes = packed_per_row * u64::from(amount);
        (
            logical_elements,
            useful_payload_bytes,
            packed_transfer_bytes,
        )
    }

    pub(crate) fn record_read(&self, hbm_type: MxDataType, dim: u32, amount: u32) {
        let (elements, useful, packed) = Self::transfer_sizes(hbm_type, dim, amount);
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        self.read_logical_elements
            .fetch_add(elements, Ordering::Relaxed);
        self.read_useful_payload_bytes
            .fetch_add(useful, Ordering::Relaxed);
        self.read_packed_transfer_bytes
            .fetch_add(packed, Ordering::Relaxed);
    }

    pub(crate) fn record_write(&self, hbm_type: MxDataType, dim: u32, amount: u32) {
        let (elements, useful, packed) = Self::transfer_sizes(hbm_type, dim, amount);
        self.write_operations.fetch_add(1, Ordering::Relaxed);
        self.write_logical_elements
            .fetch_add(elements, Ordering::Relaxed);
        self.write_useful_payload_bytes
            .fetch_add(useful, Ordering::Relaxed);
        self.write_packed_transfer_bytes
            .fetch_add(packed, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> DmaStatisticsSnapshot {
        DmaStatisticsSnapshot {
            read_operations: self.read_operations.load(Ordering::Relaxed),
            write_operations: self.write_operations.load(Ordering::Relaxed),
            read_logical_elements: self.read_logical_elements.load(Ordering::Relaxed),
            write_logical_elements: self.write_logical_elements.load(Ordering::Relaxed),
            read_useful_payload_bytes: self.read_useful_payload_bytes.load(Ordering::Relaxed),
            write_useful_payload_bytes: self.write_useful_payload_bytes.load(Ordering::Relaxed),
            read_packed_transfer_bytes: self.read_packed_transfer_bytes.load(Ordering::Relaxed),
            write_packed_transfer_bytes: self.write_packed_transfer_bytes.load(Ordering::Relaxed),
            read_coalesced_line_requests: 0,
            write_coalesced_line_requests: 0,
            read_coalesced_line_bytes: 0,
            write_coalesced_line_bytes: 0,
            read_physical_burst_bytes: 0,
            write_physical_burst_bytes: 0,
            read_physical_bursts: 0,
            write_physical_bursts: 0,
        }
    }
}

/// Derived byte-layout for one MX transfer iteration.
///
/// Computed identically for both transfer directions from the HBM data type,
/// the stride register value, and the per-iteration element count (`load_dim`
/// / `store_dim`).
struct MxLayout {
    element_ty: DataType,
    // Retained for layout assertions and calibration tests; production
    // transfers consume the already-derived byte lengths below.
    #[allow(dead_code)]
    element_bits: u8,
    /// Scale element bit-width (equals `element_bits` for non-MX types, where
    /// it is unused).
    scale_bits: u8,
    /// Packed-byte stride for the element stream.
    element_stride_bytes: u32,
    /// Packed-byte stride for the scale stream.
    scale_stride_bytes: u32,
    /// Element bytes per iteration.
    len_in_bytes: u32,
    /// Scale bytes per iteration (0 for non-MX types).
    scale_len_in_bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DmaTransferDirection {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DmaLineRequest {
    pub(crate) address: u64,
    pub(crate) fully_covered: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DmaLineManifest {
    pub(crate) reads: Vec<DmaLineRequest>,
    pub(crate) writes: Vec<DmaLineRequest>,
    pub(crate) full_lines: u64,
    pub(crate) partial_lines: u64,
}

impl MxLayout {
    fn compute(hbm_type: MxDataType, stride: u32, dim: u32, stride_unit: AddressUnit) -> Self {
        let element_ty = hbm_type.element_type();
        let element_bits = element_ty.size_in_bits();

        // Scale element bit-width (element_bits for plain types, where the
        // scale stream is unused).
        let scale_bits = match hbm_type {
            MxDataType::Mx { scale, .. } => scale.size_in_bits(),
            _ => element_bits,
        };

        assert!(element_bits.is_power_of_two());

        let stride_elements = match stride_unit {
            AddressUnit::Elements => stride,
            AddressUnit::Bytes => {
                let stride_bits = stride
                    .checked_mul(8)
                    .expect("DMA byte stride overflows bit count");
                assert!(
                    stride_bits.is_multiple_of(element_bits as u32),
                    "byte stride {stride} does not address a whole number of {element_bits}-bit elements"
                );
                stride_bits / element_bits as u32
            }
        };
        let element_stride_bytes = packed_bytes_exact(stride_elements, element_bits);

        let len_in_bits = element_bits as u32 * dim;
        // A load must be a whole number of bytes. This was previously required
        // to be a full 64-byte HBM burst (`8 * 64`); relaxed for sub-64 MLEN,
        // which packs fewer than 64 bytes per row.
        assert!(len_in_bits.is_multiple_of(8));
        let len_in_bytes = len_in_bits / 8;

        let (scale_len_in_bytes, scale_stride_bytes) = if let MxDataType::Mx {
            elem: _,
            scale,
            block,
        } = hbm_type
        {
            let scale_bits = scale.size_in_bits();
            assert!(scale_bits.is_power_of_two());
            assert!(dim.is_multiple_of(block));
            assert!(stride_elements.is_multiple_of(block));
            (
                packed_bytes_exact(dim / block, scale_bits),
                packed_bytes_exact(stride_elements / block, scale_bits),
            )
        } else {
            (0, 0)
        };

        MxLayout {
            element_ty,
            element_bits,
            scale_bits,
            element_stride_bytes,
            scale_stride_bytes,
            len_in_bytes,
            scale_len_in_bytes,
        }
    }
}

pub(crate) const fn packed_bytes(elements: u32, bits_per_element: u8) -> u32 {
    (elements * bits_per_element as u32).div_ceil(8)
}

pub(crate) fn packed_bytes_exact(elements: u32, bits_per_element: u8) -> u32 {
    let bits = elements
        .checked_mul(bits_per_element as u32)
        .expect("packed DMA byte count overflow");
    assert!(
        bits.is_multiple_of(8),
        "{elements} elements at {bits_per_element} bits do not form a byte-aligned DMA region"
    );
    bits / 8
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AddressUnit {
    /// Production ISA offsets and strides count logical tensor elements.
    Elements,
    /// Calibration drivers may provide already-packed byte strides.
    #[allow(dead_code)]
    Bytes,
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
    pub(crate) stride_unit: AddressUnit,
}

fn mx_read_chunks(
    layout: &MxLayout,
    element_base: u64,
    scale_base: u64,
    amount: u32,
) -> (usize, Vec<memory::chunked::ChunkRead>) {
    let total_element_bytes = layout.len_in_bytes as usize * amount as usize;
    let total_scale_bytes = layout.scale_len_in_bytes as usize * amount as usize;
    let mut reads = Vec::new();
    for row in 0..amount {
        let element_addr = element_base + u64::from(row * layout.element_stride_bytes);
        let scale_addr = scale_base + u64::from(row * layout.scale_stride_bytes);
        let element_offset = row as usize * layout.len_in_bytes as usize;
        let scale_offset = total_element_bytes + row as usize * layout.scale_len_in_bytes as usize;

        let element_end = element_addr + u64::from(layout.len_in_bytes);
        let mut line = (element_addr / 64) * 64;
        while line < element_end {
            let start = line.max(element_addr);
            let end = (line + 64).min(element_end);
            reads.push(memory::chunked::ChunkRead {
                addr: start,
                dst_offset: element_offset + (start - element_addr) as usize,
                len: (end - start) as usize,
            });
            line += 64;
        }

        if layout.scale_len_in_bytes > 0 {
            let scale_end = scale_addr + u64::from(layout.scale_len_in_bytes);
            let mut line = (scale_addr / 64) * 64;
            while line < scale_end {
                let start = line.max(scale_addr);
                let end = (line + 64).min(scale_end);
                reads.push(memory::chunked::ChunkRead {
                    addr: start,
                    dst_offset: scale_offset + (start - scale_addr) as usize,
                    len: (end - start) as usize,
                });
                line += 64;
            }
        }
    }
    (total_element_bytes + total_scale_bytes, reads)
}

fn mx_write_chunks(
    layout: &MxLayout,
    element_base: u64,
    scale_base: u64,
    amount: u32,
) -> Vec<memory::chunked::ChunkWrite> {
    let mut writes = Vec::with_capacity(amount as usize * 2);
    for row in 0..amount {
        writes.push(memory::chunked::ChunkWrite {
            addr: element_base + u64::from(row * layout.element_stride_bytes),
            bytes: vec![0; layout.len_in_bytes as usize],
        });
        if layout.scale_len_in_bytes > 0 {
            writes.push(memory::chunked::ChunkWrite {
                addr: scale_base + u64::from(row * layout.scale_stride_bytes),
                bytes: vec![0; layout.scale_len_in_bytes as usize],
            });
        }
    }
    writes
}

pub(crate) fn plan_mx_dma_lines(
    region: MxRegion,
    dim: u32,
    amount: u32,
    direction: DmaTransferDirection,
) -> DmaLineManifest {
    let stride = if region.rstride == 1 {
        region.stride
    } else {
        dim
    };
    let layout = MxLayout::compute(region.hbm_type, stride, dim, region.stride_unit);
    match direction {
        DmaTransferDirection::Read => {
            let (_, reads) = mx_read_chunks(&layout, region.index, region.scale_index, amount);
            let reads = memory::chunked::plan_gather(reads)
                .into_iter()
                .map(|line| DmaLineRequest {
                    address: line.address,
                    fully_covered: line.fully_covered,
                })
                .collect::<Vec<_>>();
            let full_lines = reads.iter().filter(|line| line.fully_covered).count() as u64;
            DmaLineManifest {
                partial_lines: reads.len() as u64 - full_lines,
                full_lines,
                reads,
                writes: Vec::new(),
            }
        }
        DmaTransferDirection::Write => {
            let writes = memory::chunked::plan_scatter(mx_write_chunks(
                &layout,
                region.index,
                region.scale_index,
                amount,
            ));
            let write_lines = writes
                .iter()
                .map(|line| DmaLineRequest {
                    address: line.address,
                    fully_covered: line.fully_covered,
                })
                .collect::<Vec<_>>();
            let read_lines = write_lines
                .iter()
                .copied()
                .filter(|line| !line.fully_covered)
                .collect::<Vec<_>>();
            let full_lines = write_lines.iter().filter(|line| line.fully_covered).count() as u64;
            DmaLineManifest {
                partial_lines: write_lines.len() as u64 - full_lines,
                full_lines,
                reads: read_lines,
                writes: write_lines,
            }
        }
    }
}

/// Execute only the modeled memory portion of one production DMA operation.
///
/// Host-side quantization and tensor copies do not advance the transactional
/// executor clock. Bypassing them makes calibration much faster while using
/// exactly the same packed layout and `gather`/`scatter` line planner as the
/// numerical path.
pub(crate) async fn execute_mx_dma_timing(
    hbm: &Arc<dyn ErasedMemoryModel>,
    region: MxRegion,
    dim: u32,
    amount: u32,
    direction: DmaTransferDirection,
) {
    let stride = if region.rstride == 1 {
        region.stride
    } else {
        dim
    };
    let layout = MxLayout::compute(region.hbm_type, stride, dim, region.stride_unit);
    match direction {
        DmaTransferDirection::Read => {
            let (total, reads) = mx_read_chunks(&layout, region.index, region.scale_index, amount);
            let _ = memory::chunked::gather(hbm, total, reads).await;
        }
        DmaTransferDirection::Write => {
            let _ = memory::chunked::scatter(
                hbm,
                mx_write_chunks(&layout, region.index, region.scale_index, amount),
            )
            .await;
        }
    }
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
    statistics: &Arc<DmaStatistics>,
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
        stride_unit,
    } = region;
    statistics.record_read(hbm_type, load_dim, load_amount);
    let stride = if rstride == 1 { stride } else { load_dim };
    let hbm = hbm.clone();

    Executor::current().spawn(async move {
        let layout = MxLayout::compute(hbm_type, stride, load_dim, stride_unit);
        let element_ty = layout.element_ty;
        let scale_bits = layout.scale_bits;
        let len_in_bytes_per_load = layout.len_in_bytes;
        let scale_len_in_bytes_per_load = layout.scale_len_in_bytes;

        let (total_len, reads) = mx_read_chunks(&layout, index, scale_index, load_amount);
        let gathered = memory::chunked::gather(&hbm, total_len, reads).await;
        let total_bytes = len_in_bytes_per_load as usize * load_amount as usize;
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
                &bytes[bytes_start
                    ..bytes_start + write_amount as usize * len_in_bytes_per_load as usize],
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
    statistics: &Arc<DmaStatistics>,
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
        stride_unit,
    } = region;
    statistics.record_write(hbm_type, store_dim, store_amount);
    let stride = if rstride == 1 { stride } else { store_dim };

    let layout = MxLayout::compute(hbm_type, stride, store_dim, stride_unit);
    let len_in_bytes_per_store = layout.len_in_bytes;
    let scale_len_in_bytes_per_store = layout.scale_len_in_bytes;
    let mut writes = Vec::new();

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
        let element_addr = index + u64::from(store_iter * layout.element_stride_bytes);
        let scale_addr = scale_index + u64::from(store_iter * layout.scale_stride_bytes);

        writes.push(memory::chunked::ChunkWrite {
            addr: element_addr,
            bytes: element_bytes[..len_in_bytes_per_store as usize].to_vec(),
        });

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

            writes.push(memory::chunked::ChunkWrite {
                addr: scale_addr,
                bytes: scale_bytes[..total_scale_bytes].to_vec(),
            });
        }

        tracing::debug!("[H_STORE_V] Store iter {} completed", store_iter);
    }

    // Commit every packed fragment as one DMA operation. The scatter layer
    // coalesces element and scale patches sharing a physical HBM line and
    // waits for every completion-aware Ramulator write callback.
    let logical_written = memory::chunked::scatter(hbm, writes).await;
    tracing::debug!(logical_written, "H_STORE_V coalesced write completed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::FpType;

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
        // 1, so stride_scale == stride; len = 8 * dim / 8 bytes.
        let layout = MxLayout::compute(
            MxDataType::Plain(DataType::Fp(e4m3())),
            64,
            64,
            AddressUnit::Elements,
        );
        assert_eq!(layout.element_ty, DataType::Fp(e4m3()));
        assert_eq!(layout.element_bits, 8);
        // For plain types scale_bits mirrors element_bits (the field is unused).
        assert_eq!(layout.scale_bits, 8);
        assert_eq!(layout.element_stride_bytes, 64);
        assert_eq!(layout.scale_stride_bytes, 0);
        assert_eq!(layout.len_in_bytes, 64); // 8 bits * 64 / 8
        assert_eq!(layout.scale_len_in_bytes, 0);
    }

    #[test]
    fn test_layout_mx_block_scale_stream() {
        // Mx { e4m3 elems, E8M0 scale, block 32 }: ratio = 8*32/8 = 32, so one
        // scale per 32 elements. dim 64 -> 2 scale elements -> 2 bytes (E8M0 is
        // 8-bit). stride_scale = stride / ratio.
        let ty = MxDataType::Mx {
            elem: DataType::Fp(e4m3()),
            scale: DataType::Fp(FpType::E8M0),
            block: 32,
        };
        let layout = MxLayout::compute(ty, 64, 64, AddressUnit::Elements);
        assert_eq!(layout.element_bits, 8);
        assert_eq!(layout.scale_bits, 8); // E8M0
        assert_eq!(layout.element_stride_bytes, 64);
        assert_eq!(layout.scale_stride_bytes, 2);
        assert_eq!(layout.len_in_bytes, 64); // 8 * 64 / 8
        assert_eq!(layout.scale_len_in_bytes, 2); // 8 bits * (64/32) / 8
    }

    #[test]
    fn test_layout_len_scales_with_dim() {
        // len_in_bytes is element_bits * dim / 8; halving dim halves the length.
        let plain = MxDataType::Plain(DataType::Fp(e4m3()));
        assert_eq!(
            MxLayout::compute(plain, 32, 32, AddressUnit::Elements).len_in_bytes,
            32
        );
        assert_eq!(
            MxLayout::compute(plain, 32, 128, AddressUnit::Elements).len_in_bytes,
            128
        );
    }

    #[test]
    fn test_layout_16bit_element_doubles_byte_length() {
        // F16 is 16-bit, so len_in_bytes = 16 * dim / 8 = 2 * dim.
        let plain = MxDataType::Plain(DataType::Fp(FpType::F16));
        let layout = MxLayout::compute(plain, 64, 64, AddressUnit::Elements);
        assert_eq!(layout.element_bits, 16);
        assert_eq!(layout.len_in_bytes, 128); // 16 * 64 / 8
    }

    #[test]
    fn test_layout_mxint4_uses_packed_element_offsets() {
        let ty = MxDataType::Mx {
            elem: DataType::Int(quantize::IntType { width: 4 }),
            scale: DataType::Fp(FpType::E8M0),
            block: 8,
        };
        let layout = MxLayout::compute(ty, 64, 64, AddressUnit::Elements);
        assert_eq!(layout.len_in_bytes, 32);
        assert_eq!(layout.element_stride_bytes, 32);
        assert_eq!(layout.scale_len_in_bytes, 8);
        assert_eq!(layout.scale_stride_bytes, 8);
        assert_eq!(packed_bytes(65, 4), 33);
    }

    #[test]
    fn test_byte_stride_calibration_matches_element_stride() {
        let ty = MxDataType::Plain(DataType::Int(quantize::IntType { width: 4 }));
        let from_elements = MxLayout::compute(ty, 64, 64, AddressUnit::Elements);
        let from_bytes = MxLayout::compute(ty, 32, 64, AddressUnit::Bytes);
        assert_eq!(
            from_elements.element_stride_bytes,
            from_bytes.element_stride_bytes
        );
    }

    #[test]
    fn test_mxint_payload_accounting_is_bit_packed() {
        for (width, expected_bytes) in [(2, 34), (4, 66), (8, 130)] {
            let ty = MxDataType::Mx {
                elem: DataType::Int(quantize::IntType { width }),
                scale: DataType::Fp(FpType::E8M0),
                block: 64,
            };
            let (elements, useful, packed) = DmaStatistics::transfer_sizes(ty, 64, 2);
            assert_eq!(elements, 128);
            assert_eq!(useful, expected_bytes);
            assert_eq!(packed, expected_bytes);
        }
    }

    #[test]
    fn test_mxfp_payload_accounting_includes_shared_scales() {
        let e1m2 = FpType {
            sign: true,
            exponent: 1,
            mantissa: 2,
        };
        for (format, expected_bytes) in [(e1m2, 34), (e4m3(), 66)] {
            let ty = MxDataType::Mx {
                elem: DataType::Fp(format),
                scale: DataType::Fp(FpType::E8M0),
                block: 32,
            };
            let (elements, useful, packed) = DmaStatistics::transfer_sizes(ty, 64, 1);
            assert_eq!(elements, 64);
            assert_eq!(useful, expected_bytes);
            assert_eq!(packed, expected_bytes);
        }
    }
}
