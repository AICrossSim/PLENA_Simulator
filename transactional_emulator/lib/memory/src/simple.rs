use runtime::Duration;
use tokio::sync::Semaphore;

use crate::{MemoryTimingModel, frfcfs::FrFcFs};

/// A very simple memory timing model focused on primary timing only.
///
/// Currently refresh is ignored.
pub struct SimpleTiming {
    tck: Duration,
    latency: Duration,
    xfer: Duration,
    concurrent_read: Semaphore,
    concurrent_write: Semaphore,
    cmd: Semaphore,
    data: Semaphore,
    row_width: u32,
    bank_width: u32,
    banks: Vec<FrFcFs>,
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
        let xfer_cycle = 64 * 8 / bus_width / 2;
        let num_channel = num_channel as usize;
        Self {
            tck,
            latency: tck * cas,
            xfer: tck * xfer_cycle,
            concurrent_read: Semaphore::new(16 * num_channel),
            concurrent_write: Semaphore::new(16 * num_channel),
            cmd: Semaphore::new(num_channel),
            data: Semaphore::new(num_channel),
            row_width,
            bank_width,
            banks: (0..(1 << bank_width))
                .map(|_| FrFcFs::new(tck, rcd, rp))
                .collect(),
        }
    }

    pub fn preset_ddr4_2400p(num_channel: u32) -> Self {
        let channel_width = num_channel.ilog2();
        Self::new(
            Duration::from_picos(833),
            16,
            16,
            16,
            64,
            // 8GBx8
            // For now, for channel simulation, map it to banks..
            16 - channel_width,
            4 + channel_width,
            num_channel,
        )
    }
}

impl MemoryTimingModel for SimpleTiming {
    async fn read(&self, addr: u64) {
        let rt = runtime::Executor::current();

        let _permit = self.concurrent_read.acquire().await.unwrap();

        // Send the command.
        let cmd = self.cmd.acquire().await.unwrap();
        rt.resolve_at(self.tck).await;
        drop(cmd);

        let bank_id = ((addr >> 6) & ((1 << self.bank_width) - 1)) as usize;
        let row_id = ((addr >> (10 + self.bank_width)) & ((1 << self.row_width) - 1)) as u32;
        let row_permit = self.banks[bank_id].acquire(row_id).await;

        // Wait for CAS delay.
        rt.resolve_at(self.latency).await;
        drop(row_permit);

        let _data = self.data.acquire().await.unwrap();
        rt.resolve_at(self.xfer).await;
    }

    async fn write(&self, addr: u64) {
        let rt = runtime::Executor::current();

        let _permit = self.concurrent_write.acquire().await.unwrap();

        // Send the command.
        let cmd = self.cmd.acquire().await.unwrap();
        rt.resolve_at(self.tck).await;
        drop(cmd);

        let bank_id = ((addr >> 10) & ((1 << self.bank_width) - 1)) as usize;
        let row_id = ((addr >> (10 + self.bank_width)) & ((1 << self.row_width) - 1)) as u32;
        let row_permit = self.banks[bank_id].acquire(row_id).await;
        drop(row_permit);

        // Send the data
        let _data = self.data.acquire().await.unwrap();
        rt.resolve_at(self.xfer).await;
    }
}
