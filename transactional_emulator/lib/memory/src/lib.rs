pub mod chunked;
mod frfcfs;
mod naive;
mod simple;
pub mod testutils;

use std::mem::ManuallyDrop;
use std::sync::Mutex;

pub use naive::NaiveTiming;
pub use simple::SimpleTiming;

#[derive(Copy, Clone)]
pub struct Statistics {
    /// Number of 64-byte requests presented through the [`MemoryModel`] API.
    pub read_requests: u64,
    /// Number of 64-byte requests presented through the [`MemoryModel`] API.
    pub write_requests: u64,
    pub total_bytes_read: u64,
    pub total_bytes_written: u64,
}

#[trait_variant::make(Send)]
pub trait MemoryTimingModel: Send + Sync {
    /// Read 64-bytes of memory.
    ///
    /// We fix to 64-bytes to accomodate memory emulators.
    async fn read(&self, addr: u64);

    /// Write 64-bytes of memory.
    async fn write(&self, addr: u64);
}

impl<T: MemoryTimingModel> MemoryTimingModel for ManuallyDrop<T> {
    async fn read(&self, addr: u64) {
        T::read(self, addr).await
    }

    async fn write(&self, addr: u64) {
        T::write(self, addr).await
    }
}

#[trait_variant::make(Send)]
pub trait MemoryModel: Send + Sync {
    /// Read 64-bytes of memory.
    async fn read(&self, addr: u64) -> [u8; 64];

    /// Write 64-bytes of memory.
    async fn write(&self, addr: u64, bytes: [u8; 64]);
}

#[async_trait::async_trait]
pub trait ErasedMemoryModel: Send + Sync {
    async fn box_read(&self, addr: u64) -> [u8; 64];
    async fn box_write(&self, addr: u64, bytes: [u8; 64]);
}

#[async_trait::async_trait]
impl<T: MemoryModel> ErasedMemoryModel for T {
    async fn box_read(&self, addr: u64) -> [u8; 64] {
        self.read(addr).await
    }

    async fn box_write(&self, addr: u64, bytes: [u8; 64]) {
        self.write(addr, bytes).await
    }
}

impl MemoryModel for dyn ErasedMemoryModel {
    async fn read(&self, addr: u64) -> [u8; 64] {
        self.box_read(addr).await
    }

    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        self.box_write(addr, bytes).await
    }
}

/// A memory that discards all written data.
///
/// This is useful to just test the timing without caring the actual data.
pub struct NoData;

impl MemoryModel for NoData {
    /// Read 64-bytes of memory.
    async fn read(&self, _addr: u64) -> [u8; 64] {
        [0; 64]
    }

    /// Write 64-bytes of memory.
    async fn write(&self, _addr: u64, _bytes: [u8; 64]) {}
}

/// A simulated memory that is backed by memory.
///
/// This is useful to just test the timing without caring the actual data.
pub struct MemoryBacked {
    data: Mutex<Vec<[u8; 64]>>,
}

impl MemoryBacked {
    pub fn with_capacity(size: usize) -> Self {
        assert!(size.is_multiple_of(64));
        Self {
            data: Mutex::new(vec![[0; 64]; size / 64]),
        }
    }

    pub fn with_data(&self, f: impl FnOnce(&mut [u8])) {
        use zerocopy::IntoBytes;

        let mut guard = self.data.lock().unwrap();
        f(guard.as_mut_bytes())
    }
}

impl MemoryModel for MemoryBacked {
    /// Read 64-bytes of memory.
    async fn read(&self, addr: u64) -> [u8; 64] {
        // HBM bursts read aligned 64-byte words covering [addr, addr+len); an
        // H_PREFETCH that requests more rows than a tensor has (e.g. blen > seq_len)
        // over-reads past the tensor's end. Those bytes land in unused (padding) VRAM
        // rows and are never compared, so return zeros for out-of-capacity addresses
        // instead of panicking on the backing Vec index.
        self.data
            .lock()
            .unwrap()
            .get(addr as usize / 64)
            .copied()
            .unwrap_or([0u8; 64])
    }

    /// Write 64-bytes of memory.
    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        self.data.lock().unwrap()[addr as usize / 64] = bytes;
    }
}

/// Combine a data model with an extra timing model.
pub struct WithTiming<T, M> {
    timing: T,
    data: M,
}

impl<T, M> WithTiming<T, M> {
    pub fn new(timing: T, data: M) -> Self {
        WithTiming { timing, data }
    }

    pub fn data(&self) -> &M {
        &self.data
    }

    pub fn timing(&self) -> &T {
        &self.timing
    }
}

impl<T: MemoryTimingModel, M: MemoryModel> MemoryModel for WithTiming<T, M> {
    /// Read 64-bytes of memory.
    async fn read(&self, addr: u64) -> [u8; 64] {
        self.timing.read(addr).await;
        self.data.read(addr).await
    }

    /// Write 64-bytes of memory.
    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        self.timing.write(addr).await;
        self.data.write(addr, bytes).await
    }
}

// Memory model with utilization statistics
pub struct WithStats<T> {
    model: T,
    statistics: Mutex<Statistics>,
}

impl<T> WithStats<T> {
    pub fn new(model: T) -> Self {
        let stats = Statistics {
            read_requests: 0,
            write_requests: 0,
            total_bytes_read: 0,
            total_bytes_written: 0,
        };
        WithStats {
            model,
            statistics: Mutex::new(stats),
        }
    }

    pub fn model(&self) -> &T {
        &self.model
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics.lock().unwrap().clone()
    }
}

impl<T: MemoryModel> MemoryModel for WithStats<T> {
    async fn read(&self, addr: u64) -> [u8; 64] {
        {
            let mut guard = self.statistics.lock().unwrap();
            guard.read_requests += 1;
            guard.total_bytes_read += 64;
        }
        self.model.read(addr).await
    }

    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        {
            let mut guard = self.statistics.lock().unwrap();
            guard.write_requests += 1;
            guard.total_bytes_written += 64;
        }
        self.model.write(addr, bytes).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_backed_roundtrip() {
        let mb = MemoryBacked::with_capacity(128);
        let mut block = [0u8; 64];
        for (i, b) in block.iter_mut().enumerate() {
            *b = i as u8;
        }
        mb.write(64, block).await;
        assert_eq!(mb.read(64).await, block);
        assert_eq!(mb.read(0).await, [0u8; 64]); // other block untouched
    }

    #[tokio::test]
    async fn test_nodata_discards_writes() {
        let nd = NoData;
        nd.write(0, [7u8; 64]).await;
        assert_eq!(nd.read(0).await, [0u8; 64]); // always reads back zero
    }

    #[tokio::test]
    async fn test_with_stats_counts_bytes_and_delegates() {
        let s = WithStats::new(MemoryBacked::with_capacity(128));
        let _ = s.read(0).await;
        let _ = s.read(64).await;
        s.write(0, [1u8; 64]).await;

        let stats = s.statistics();
        assert_eq!(stats.read_requests, 2);
        assert_eq!(stats.write_requests, 1);
        assert_eq!(stats.total_bytes_read, 128); // two 64-byte reads
        assert_eq!(stats.total_bytes_written, 64); // one 64-byte write
        assert_eq!(s.read(0).await, [1u8; 64]); // data still flows through
    }

    /// A zero-cost timing model used only to exercise `WithTiming`'s data path
    /// without involving the simulation executor.
    struct NoTiming;

    impl MemoryTimingModel for NoTiming {
        async fn read(&self, _addr: u64) {}
        async fn write(&self, _addr: u64) {}
    }

    #[tokio::test]
    async fn test_with_timing_delegates_data() {
        let wt = WithTiming::new(NoTiming, MemoryBacked::with_capacity(64));
        wt.write(0, [9u8; 64]).await;
        assert_eq!(wt.read(0).await, [9u8; 64]);
        assert_eq!(wt.data().read(0).await, [9u8; 64]); // data() accessor
    }
}
