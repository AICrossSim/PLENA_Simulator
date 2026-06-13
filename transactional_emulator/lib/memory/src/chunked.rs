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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBacked;
    use proptest::prelude::*;
    use std::sync::Arc;

    /// A `MemoryBacked` HBM seeded by `init`, returned both as a typed handle
    /// (for inspection) and as the erased `Arc` the primitives consume.
    fn seeded(
        cap: usize,
        init: impl FnOnce(&mut [u8]),
    ) -> (Arc<MemoryBacked>, Arc<dyn ErasedMemoryModel>) {
        let mb = Arc::new(MemoryBacked::with_capacity(cap));
        mb.with_data(init);
        let hbm: Arc<dyn ErasedMemoryModel> = mb.clone();
        (mb, hbm)
    }

    #[tokio::test]
    async fn test_gather_single_aligned_block() {
        let (_mb, hbm) = seeded(128, |b| {
            for (i, byte) in b[..64].iter_mut().enumerate() {
                *byte = i as u8;
            }
        });
        let out = gather(
            &hbm,
            64,
            vec![ChunkRead {
                addr: 0,
                dst_offset: 0,
                len: 64,
            }],
        )
        .await;
        assert_eq!(out, (0..64).map(|i| i as u8).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn test_gather_unaligned_slice_within_block() {
        let (_mb, hbm) = seeded(128, |b| {
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = i as u8;
            }
        });
        // 10 bytes from addr 70 -> block [64, 128), within-offset 6.
        let out = gather(
            &hbm,
            10,
            vec![ChunkRead {
                addr: 70,
                dst_offset: 0,
                len: 10,
            }],
        )
        .await;
        assert_eq!(out, (70u8..80).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn test_gather_places_each_read_at_its_offset() {
        let (_mb, hbm) = seeded(128, |b| {
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = i as u8;
            }
        });
        let out = gather(
            &hbm,
            8,
            vec![
                ChunkRead {
                    addr: 0,
                    dst_offset: 4,
                    len: 4,
                },
                ChunkRead {
                    addr: 64,
                    dst_offset: 0,
                    len: 4,
                },
            ],
        )
        .await;
        assert_eq!(out, vec![64, 65, 66, 67, 0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_gather_clamps_read_to_block_end() {
        let (_mb, hbm) = seeded(128, |b| {
            for (i, byte) in b.iter_mut().enumerate() {
                *byte = i as u8;
            }
        });
        // addr 60, len 20 would cross the block boundary -> only 60..64 returned.
        let out = gather(
            &hbm,
            20,
            vec![ChunkRead {
                addr: 60,
                dst_offset: 0,
                len: 20,
            }],
        )
        .await;
        let mut expected = vec![0u8; 20];
        expected[0..4].copy_from_slice(&[60, 61, 62, 63]);
        assert_eq!(out, expected);
    }

    #[tokio::test]
    async fn test_write_aligned_zero_pads_short_payload() {
        let (mb, hbm) = seeded(128, |_| {});
        let src: Vec<u8> = (1u8..=100).collect();
        write_aligned(&hbm, 0, 128, &src).await;
        mb.with_data(|b| {
            assert_eq!(&b[0..100], &src[..]);
            assert_eq!(&b[100..128], &[0u8; 28]);
        });
    }

    #[tokio::test]
    async fn test_write_unaligned_rmw_preserves_neighbors() {
        let (mb, hbm) = seeded(128, |b| b.fill(0xAA));
        // Write four bytes straddling the 64-byte boundary at addr 62.
        let written = write_unaligned(&hbm, 62, 4, &[1, 2, 3, 4]).await;
        assert_eq!(written, 4);
        mb.with_data(|b| {
            assert_eq!(b[61], 0xAA);
            assert_eq!(&b[62..66], &[1, 2, 3, 4]);
            assert_eq!(b[66], 0xAA);
        });
    }

    #[tokio::test]
    async fn test_write_unaligned_stops_when_src_exhausted() {
        let (mb, hbm) = seeded(128, |b| b.fill(0xAA));
        let written = write_unaligned(&hbm, 0, 64, &[1, 2]).await; // total_len 64 > src len 2
        assert_eq!(written, 2);
        mb.with_data(|b| {
            assert_eq!(&b[0..2], &[1, 2]);
            assert_eq!(b[2], 0xAA);
        });
    }

    proptest! {
        /// `gather` over an arbitrary read list must equal a straightforward
        /// per-read assembly against the same backing bytes.
        #[test]
        fn prop_gather_matches_naive_assembly(
            seed in prop::collection::vec(any::<u8>(), 256),
            reads in prop::collection::vec((0u64..256, 0usize..=64), 0..12),
        ) {
            let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();

            let mb = Arc::new(MemoryBacked::with_capacity(256));
            mb.with_data(|b| b.copy_from_slice(&seed));
            let hbm: Arc<dyn ErasedMemoryModel> = mb.clone();

            let total_len: usize = reads.iter().map(|(_, len)| *len).sum();
            let mut naive = vec![0u8; total_len];
            let mut chunk_reads = Vec::new();
            let mut dst = 0usize;
            for (addr, len) in &reads {
                let within = (*addr % 64) as usize;
                let n = (*len).min(64 - within);
                naive[dst..dst + n].copy_from_slice(&seed[*addr as usize..*addr as usize + n]);
                chunk_reads.push(ChunkRead { addr: *addr, dst_offset: dst, len: *len });
                dst += *len;
            }

            let got = rt.block_on(gather(&hbm, total_len, chunk_reads));
            prop_assert_eq!(got, naive);
        }
    }
}
