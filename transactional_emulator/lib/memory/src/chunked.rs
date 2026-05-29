//! Strided, chunked byte-transfer primitives over a [`MemoryModel`].
//!
//! These operate purely on bytes — they have no knowledge of MX formats,
//! tensors, or SRAM. Higher layers (the accelerator's `dma` module) build
//! the MX-aware transfer logic on top of these.
//!
//! All reads/writes go through the 64-byte-granular [`MemoryModel`] API, so
//! every primitive here works in terms of 64-byte aligned blocks: reads of
//! unaligned ranges fetch the containing aligned block and slice; writes to
//! unaligned ranges read-modify-write the containing blocks.

use std::sync::Arc;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;

use crate::{ErasedMemoryModel, MemoryModel};

/// A single chunked read request.
///
/// Reads `len` bytes starting at `addr` (which need not be 64-aligned: the
/// model fetches the containing aligned 64-byte block and slices out
/// `[addr % 64 .. addr % 64 + len]`), depositing them at `dst_offset` in the
/// gather buffer.
pub struct ChunkRead {
    pub addr: u64,
    pub dst_offset: usize,
    pub len: usize,
}

/// Issue every [`ChunkRead`] concurrently against `hbm` and assemble the
/// results into a single `total_len`-byte buffer.
///
/// All reads race in one pool, preserving the simulator's concurrent-access
/// timing; the buffer is filled as each read completes (completion order does
/// not matter — each result carries its own destination offset).
pub async fn gather(
    hbm: &Arc<dyn ErasedMemoryModel>,
    total_len: usize,
    reads: Vec<ChunkRead>,
) -> Vec<u8> {
    let mut out = vec![0u8; total_len];
    let mut futures = FuturesUnordered::new();
    for r in reads {
        // A single read cannot span more than the 64-byte block it lands in.
        debug_assert!(r.len <= 64, "ChunkRead::len {} exceeds 64", r.len);
        let hbm = hbm.clone();
        futures.push(async move {
            let aligned = (r.addr / 64) * 64;
            let within = (r.addr % 64) as usize;
            let block = hbm.read(aligned).await;
            let end = std::cmp::min(within + r.len, 64);
            let n = end - within;
            let mut buf = [0u8; 64];
            buf[..n].copy_from_slice(&block[within..end]);
            (r.dst_offset, buf, n)
        });
    }
    while let Some((offset, data, n)) = futures.next().await {
        out[offset..offset + n].copy_from_slice(&data[..n]);
    }
    out
}

/// Write `total_len` bytes to HBM starting at `base` (must be 64-aligned) as
/// sequential 64-byte chunks, sourcing from `src` and zero-padding where
/// `src` is shorter than the chunk range.
///
/// `total_len` (not `src.len()`) drives the chunk count, so callers can write
/// a fixed-size region from a possibly-shorter payload.
pub async fn write_aligned(
    hbm: &Arc<dyn ErasedMemoryModel>,
    base: u64,
    total_len: usize,
    src: &[u8],
) {
    for i in 0..total_len.div_ceil(64) {
        let chunk_offset = i * 64;
        let chunk_size = std::cmp::min(64, total_len - chunk_offset);
        let addr = base + (i * 64) as u64;
        assert!(addr.is_multiple_of(64));

        let mut chunk = [0u8; 64];
        if chunk_offset < src.len() {
            let copy_len = std::cmp::min(chunk_size, src.len() - chunk_offset);
            chunk[..copy_len].copy_from_slice(&src[chunk_offset..chunk_offset + copy_len]);
        }
        hbm.write(addr, chunk).await;
    }
}

/// Write up to `total_len` bytes of `src` to HBM starting at `addr` (which
/// need not be 64-aligned), via read-modify-write of the containing 64-byte
/// blocks. Stops early if `src` is exhausted. Returns the number of bytes
/// actually written.
pub async fn write_unaligned(
    hbm: &Arc<dyn ErasedMemoryModel>,
    addr: u64,
    total_len: usize,
    src: &[u8],
) -> usize {
    let mut written = 0;
    while written < total_len {
        let cur = addr + written as u64;
        let aligned = (cur / 64) * 64;
        let within = (cur % 64) as usize;

        let mut chunk = hbm.read(aligned).await;

        let remaining = total_len - written;
        let in_chunk = std::cmp::min(64 - within, remaining);
        let to_copy = std::cmp::min(in_chunk, src.len() - written);

        if to_copy > 0 {
            chunk[within..within + to_copy].copy_from_slice(&src[written..written + to_copy]);
            hbm.write(aligned, chunk).await;
            written += to_copy;
        } else {
            break;
        }
    }
    written
}
