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
        self.data.lock().unwrap()[addr as usize / 64]
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
            guard.total_bytes_read += 64;
        }
        self.model.read(addr).await
    }

    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        {
            let mut guard = self.statistics.lock().unwrap();
            guard.total_bytes_written += 64;
        }
        self.model.write(addr, bytes).await
    }
}
