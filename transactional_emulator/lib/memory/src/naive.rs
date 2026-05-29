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

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::{Executor, Instant};
    use std::sync::Arc;

    /// Drive `addrs` as fully-sequential reads on a fresh executor and return
    /// the final simulated instant. With a single task no two timers are ever
    /// pending at once, so the result is fully deterministic.
    async fn seq_reads<M: MemoryTimingModel + 'static>(model: M, addrs: &'static [u64]) -> Instant {
        let model = Arc::new(model);
        let ex = Executor::new();
        ex.spawn(async move {
            for &a in addrs {
                model.read(a).await;
            }
        });
        ex.enter(Instant::ETERNITY).await;
        ex.now()
    }

    #[tokio::test]
    async fn test_naive_sequential_read_latency() {
        // ddr4_2400p(1): tck = 833ps, CAS = 16 -> latency = 13328ps,
        // bus = 64 -> xfer_cycle = 4 -> xfer = 3332ps. One read therefore costs
        // tck + latency + xfer = 17493ps, and sequential reads simply add up.
        let now = seq_reads(NaiveTiming::preset_ddr4_2400p(1), &[0, 64, 128, 192]).await;
        assert_eq!(now, Instant::INIT + Duration::from_picos(17493) * 4u32);
    }
}
