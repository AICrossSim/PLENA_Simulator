use runtime::Duration;
use tokio::sync::Semaphore;

use crate::{MemoryTimingModel, frfcfs::FrFcFs};

struct Channel {
    concurrent_read: Semaphore,
    concurrent_write: Semaphore,
    cmd: Semaphore,
    data: Semaphore,
    banks: Vec<FrFcFs>,
}

/// A very simple memory timing model focused on primary timing only.
///
/// Currently refresh is ignored.
pub struct SimpleTiming {
    tck: Duration,
    latency: Duration,
    xfer: Duration,
    channel_width: u32,
    row_width: u32,
    bank_width: u32,
    channels: Vec<Channel>,
}

impl SimpleTiming {
    /// Create a new naive DDR timing model with given clock frequency, CAS delay, and bus width (in bits).
    pub fn new(
        tck: Duration,
        cas: u32,
        rcd: u32,
        rp: u32,
        bus_width: u32,
        row_width: u32,
        bank_width: u32,
        num_channel: u32,
    ) -> Self {
        assert!(num_channel.is_power_of_two());
        let xfer_cycle = 64 * 8 / bus_width / 2;
        let num_channel = num_channel as usize;
        let channel_width = num_channel.ilog2();
        Self {
            tck,
            latency: tck * cas,
            xfer: tck * xfer_cycle,
            channel_width,
            row_width,
            bank_width,
            channels: (0..num_channel)
                .map(|_| Channel {
                    concurrent_read: Semaphore::new(16),
                    concurrent_write: Semaphore::new(16),
                    cmd: Semaphore::new(1),
                    data: Semaphore::new(1),
                    banks: (0..(1 << bank_width))
                        .map(|_| FrFcFs::new(tck, rcd, rp))
                        .collect(),
                })
                .collect(),
        }
    }

    pub fn preset_ddr4_2400p(num_channel: u32) -> Self {
        Self::new(
            Duration::from_picos(833),
            16,
            16,
            16,
            64,
            16,
            4,
            num_channel,
        )
    }

    fn decode_addr(&self, addr: u64) -> (usize, usize, u32) {
        let addr = addr >> 6;
        let channel_id = (addr & ((1 << self.channel_width) - 1)) as usize;
        let addr = addr >> self.channel_width;
        let bank_id = (addr & ((1 << self.bank_width) - 1)) as usize;
        let addr = addr >> self.bank_width;
        let row_id = (addr & ((1 << self.row_width) - 1)) as u32;
        (channel_id, bank_id, row_id)
    }
}

impl MemoryTimingModel for SimpleTiming {
    async fn read(&self, addr: u64) {
        let rt = runtime::Executor::current();

        let (channel_id, bank_id, row_id) = self.decode_addr(addr);
        let channel = &self.channels[channel_id];

        let _permit = channel.concurrent_read.acquire().await.unwrap();

        // Send the command.
        let cmd = channel.cmd.acquire().await.unwrap();
        rt.resolve_at(self.tck).await;
        drop(cmd);

        let row_permit = channel.banks[bank_id].acquire(row_id).await;

        // Wait for CAS delay.
        rt.resolve_at(self.latency).await;
        drop(row_permit);

        let _data = channel.data.acquire().await.unwrap();
        rt.resolve_at(self.xfer).await;
    }

    async fn write(&self, addr: u64) {
        let rt = runtime::Executor::current();

        let (channel_id, bank_id, row_id) = self.decode_addr(addr);
        let channel = &self.channels[channel_id];

        let _permit = channel.concurrent_write.acquire().await.unwrap();

        // Send the command.
        let cmd = channel.cmd.acquire().await.unwrap();
        rt.resolve_at(self.tck).await;
        drop(cmd);

        let row_permit = channel.banks[bank_id].acquire(row_id).await;
        drop(row_permit);

        // Send the data
        let _data = channel.data.acquire().await.unwrap();
        rt.resolve_at(self.xfer).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryTimingModel;
    use runtime::{Duration, Executor, Instant};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_simple_single_read_latency() {
        // One read against an initially-idle bank (row activation + CAS +
        // transfer, plus the trailing precharge) has deterministic timing.
        // Pins the FR-FCFS-backed SimpleTiming latency for a single access.
        let model = Arc::new(SimpleTiming::preset_ddr4_2400p(1));
        let ex = Executor::new();
        ex.spawn(async move {
            model.read(0).await;
        });
        ex.enter(Instant::ETERNITY).await;
        assert_eq!(ex.now(), Instant::INIT + Duration::from_picos(40817));
    }
}
