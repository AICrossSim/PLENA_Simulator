use runtime::Duration;
use tokio::sync::Semaphore;

use crate::MemoryTimingModel;

/// A naive memory timing model that assumes a fixed throughput and latency.
pub struct NaiveTiming {
    tck: Duration,
    latency: Duration,
    xfer: Duration,
    concurrent_read: Semaphore,
    concurrent_write: Semaphore,
    cmd: Semaphore,
    data: Semaphore,
}

impl NaiveTiming {
    /// Create a new naive DDR timing model with given clock frequency, CAS delay, and bus width (in bits).
    pub fn new(tck: Duration, cas: u32, bus_width: u32, num_channel: u32) -> Self {
        let xfer_cycle = 64 * 8 / bus_width / 2;
        Self {
            tck,
            latency: tck * cas,
            xfer: tck * xfer_cycle,
            concurrent_read: Semaphore::new(16),
            concurrent_write: Semaphore::new(16),
            cmd: Semaphore::new(num_channel as usize),
            data: Semaphore::new(num_channel as usize),
        }
    }

    pub fn preset_ddr4_2400p(num_channel: u32) -> Self {
        Self::new(Duration::from_picos(833), 16, 64, num_channel)
    }
}

impl MemoryTimingModel for NaiveTiming {
    async fn read(&self, _addr: u64) {
        let rt = runtime::Executor::current();

        let _permit = self.concurrent_read.acquire().await.unwrap();

        // Send the command.
        let cmd = self.cmd.acquire().await.unwrap();
        rt.resolve_at(self.tck).await;
        drop(cmd);

        // Wait for CAS delay.
        rt.resolve_at(self.latency).await;

        let _data = self.data.acquire().await.unwrap();
        rt.resolve_at(self.xfer).await;
    }

    async fn write(&self, _addr: u64) {
        let rt = runtime::Executor::current();

        let _permit = self.concurrent_write.acquire().await.unwrap();

        // Send the command.
        let cmd = self.cmd.acquire().await.unwrap();
        rt.resolve_at(self.tck).await;
        drop(cmd);

        // Send the data
        let _data = self.data.acquire().await.unwrap();
        rt.resolve_at(self.xfer).await;
    }
}
