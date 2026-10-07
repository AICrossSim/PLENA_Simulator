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

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::future::join_all;

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
/// Reads are first issued in the input order, then allowed to complete
/// concurrently. Completion order does not matter because each result carries
/// its own destination offset, but deterministic issue order is part of the
/// simulator timing contract.
pub async fn gather(
    hbm: &Arc<dyn ErasedMemoryModel>,
    total_len: usize,
    reads: Vec<ChunkRead>,
) -> Vec<u8> {
    let window = std::env::var("PLENA_DMA_READ_WINDOW")
        .ok()
        .map(|v| v.parse::<usize>().expect("invalid DMA read window"));
    gather_with_window(hbm, total_len, reads, window).await
}

/// Conservative finite-credit DMA: drain one batch before admitting the next.
/// Each credit owns one 64-byte response slot and destination metadata. This
/// is deliberately not a continuously replenished credit engine. None keeps
/// the historical gather service for compatibility experiments.
pub async fn gather_with_window(
    hbm: &Arc<dyn ErasedMemoryModel>,
    total_len: usize,
    reads: Vec<ChunkRead>,
    window: Option<usize>,
) -> Vec<u8> {
    if let Some(w) = window {
        assert!((1..=64).contains(&w));
    }
    let mut out = vec![0u8; total_len];
    let batch = window.unwrap_or(reads.len().max(1));
    for group in reads.chunks(batch) {
        let futures = group.iter().map(|r| {
            // A single read cannot span more than the 64-byte block it lands in.
            debug_assert!(r.len <= 64, "ChunkRead::len {} exceeds 64", r.len);
            let hbm = hbm.clone();
            async move {
                let aligned = (r.addr / 64) * 64;
                let within = (r.addr % 64) as usize;
                let block = hbm.read(aligned).await;
                let end = std::cmp::min(within + r.len, 64);
                let n = end - within;
                let mut buf = [0u8; 64];
                buf[..n].copy_from_slice(&block[within..end]);
                (r.dst_offset, buf, n)
            }
        });
        for (offset, data, n) in join_all(futures).await {
            out[offset..offset + n].copy_from_slice(&data[..n]);
        }
    }
    out
}

/// Like [`gather`], but issue at most one physical 64-byte read for each
/// aligned HBM address in the batch.
///
/// MX scale metadata is smaller than a burst. Several logical scale slices
/// can therefore share one 64-byte line; issuing one request per slice wastes
/// bandwidth without returning any additional data. This helper models a DMA
/// burst coalescer while preserving first-occurrence issue order and the
/// original scatter order.
pub async fn gather_coalesced(
    hbm: &Arc<dyn ErasedMemoryModel>,
    total_len: usize,
    reads: Vec<ChunkRead>,
) -> Vec<u8> {
    let mut unique_addresses = Vec::new();
    let mut address_to_index = HashMap::new();

    for read in &reads {
        debug_assert!(read.len <= 64, "ChunkRead::len {} exceeds 64", read.len);
        let aligned = (read.addr / 64) * 64;
        if !address_to_index.contains_key(&aligned) {
            let index = unique_addresses.len();
            unique_addresses.push(aligned);
            address_to_index.insert(aligned, index);
        }
    }

    let futures = unique_addresses.iter().copied().map(|aligned| {
        let hbm = hbm.clone();
        async move { hbm.read(aligned).await }
    });
    let blocks = join_all(futures).await;

    let mut out = vec![0u8; total_len];
    for read in reads {
        let aligned = (read.addr / 64) * 64;
        let block = &blocks[address_to_index[&aligned]];
        let within = (read.addr % 64) as usize;
        let end = std::cmp::min(within + read.len, 64);
        let n = end - within;
        out[read.dst_offset..read.dst_offset + n].copy_from_slice(&block[within..end]);
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

/// Bounded burst write for the shared DMA experiment. Full blocks require no
/// read. Partial first/last blocks retain RMW and untouched neighbour bytes.
/// Requests are issued deterministically; the memory model supplies queue
/// backpressure. `window` is an explicit model assumption, not extra SRAM.
pub async fn write_bursted(
    hbm: &Arc<dyn ErasedMemoryModel>,
    addr: u64,
    total_len: usize,
    src: &[u8],
    window: usize,
) -> usize {
    assert!((1..=64).contains(&window));
    let count = total_len.min(src.len());
    if count == 0 {
        return 0;
    }
    let start = addr / 64 * 64;
    let end = addr + count as u64;
    let blocks = ((end - start) as usize).div_ceil(64);
    for first in (0..blocks).step_by(window) {
        let requests = (first..(first + window).min(blocks)).map(|i| async move {
            let base = start + (i * 64) as u64;
            let lo = base.max(addr);
            let hi = (base + 64).min(end);
            let mut block = if lo == base && hi == base + 64 {
                [0; 64]
            } else {
                hbm.read(base).await
            };
            block[(lo - base) as usize..(hi - base) as usize]
                .copy_from_slice(&src[(lo - addr) as usize..(hi - addr) as usize]);
            hbm.write(base, block).await;
        });
        join_all(requests).await;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBacked;
    use proptest::prelude::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn bounded_gather_limits_responses_and_preserves_destination_tags() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Outstanding {
            active: AtomicUsize,
            peak: AtomicUsize,
        }
        impl MemoryModel for Outstanding {
            async fn read(&self, addr: u64) -> [u8; 64] {
                let n = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(n, Ordering::SeqCst);
                for _ in 0..(4 - (addr / 64) % 3) {
                    tokio::task::yield_now().await;
                }
                self.active.fetch_sub(1, Ordering::SeqCst);
                [addr as u8 / 64; 64]
            }
            async fn write(&self, _addr: u64, _data: [u8; 64]) {
                panic!("gather must never issue writes")
            }
            async fn functional_read(&self, addr: u64) -> [u8; 64] {
                [addr as u8 / 64; 64]
            }
            async fn functional_write(&self, _addr: u64, _data: [u8; 64]) {
                panic!("gather must never issue writes")
            }
        }
        let tracked = Arc::new(Outstanding {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let hbm: Arc<dyn ErasedMemoryModel> = tracked.clone();
        let reads = (0..4)
            .map(|i| ChunkRead {
                addr: i * 64,
                dst_offset: (3 - i) as usize * 64,
                len: 64,
            })
            .collect();
        let out = gather_with_window(&hbm, 256, reads, Some(3)).await;
        assert_eq!(tracked.peak.load(Ordering::SeqCst), 3);
        assert_eq!(tracked.active.load(Ordering::SeqCst), 0);
        for i in 0..4 {
            assert_eq!(&out[i * 64..(i + 1) * 64], &[3 - i as u8; 64]);
        }
    }

    #[tokio::test]
    async fn burst_write_preserves_edges_without_reading_full_blocks() {
        for (addr, count, reads) in [(0, 256, 0), (3, 190, 128), (64, 64, 0), (65, 2, 64)] {
            let backing = MemoryBacked::with_capacity(512);
            backing.with_data(|b| b.fill(0xa5));
            let stats = Arc::new(crate::WithStats::new(backing));
            let hbm: Arc<dyn ErasedMemoryModel> = stats.clone();
            let source = vec![0x37; count];
            assert_eq!(write_bursted(&hbm, addr, count, &source, 16).await, count);
            assert_eq!(stats.statistics().total_bytes_read, reads);
            stats.model().with_data(|b| {
                assert!(b[..addr as usize].iter().all(|&v| v == 0xa5));
                assert_eq!(&b[addr as usize..addr as usize + count], &source);
                assert!(b[addr as usize + count..].iter().all(|&v| v == 0xa5));
            });
        }
    }

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

    struct RecordingMemory {
        read_order: Mutex<Vec<u64>>,
    }

    impl MemoryModel for RecordingMemory {
        async fn read(&self, addr: u64) -> [u8; 64] {
            self.read_order.lock().unwrap().push(addr);
            [0; 64]
        }

        async fn write(&self, _addr: u64, _bytes: [u8; 64]) {}
        async fn functional_read(&self, _addr: u64) -> [u8; 64] {
            [0; 64]
        }
        async fn functional_write(&self, _addr: u64, _bytes: [u8; 64]) {}
    }

    #[tokio::test]
    async fn test_gather_issues_reads_in_input_order() {
        let memory = Arc::new(RecordingMemory {
            read_order: Mutex::new(Vec::new()),
        });
        let hbm: Arc<dyn ErasedMemoryModel> = memory.clone();
        let _ = gather(
            &hbm,
            12,
            vec![
                ChunkRead {
                    addr: 128,
                    dst_offset: 0,
                    len: 4,
                },
                ChunkRead {
                    addr: 0,
                    dst_offset: 4,
                    len: 4,
                },
                ChunkRead {
                    addr: 64,
                    dst_offset: 8,
                    len: 4,
                },
            ],
        )
        .await;
        assert_eq!(*memory.read_order.lock().unwrap(), vec![128, 0, 64]);
    }

    #[tokio::test]
    async fn test_gather_coalesced_reads_shared_block_once_and_scatters_slices() {
        let memory = Arc::new(RecordingMemory {
            read_order: Mutex::new(Vec::new()),
        });
        let hbm: Arc<dyn ErasedMemoryModel> = memory.clone();

        let out = gather_coalesced(
            &hbm,
            12,
            vec![
                ChunkRead {
                    addr: 72,
                    dst_offset: 0,
                    len: 4,
                },
                ChunkRead {
                    addr: 0,
                    dst_offset: 4,
                    len: 4,
                },
                ChunkRead {
                    addr: 80,
                    dst_offset: 8,
                    len: 4,
                },
            ],
        )
        .await;

        assert_eq!(out, vec![0; 12]);
        assert_eq!(*memory.read_order.lock().unwrap(), vec![64, 0]);
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
