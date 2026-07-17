use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use runtime::{Duration, Executor, Instant};

use crate::raw::Ramulator as RawRamulator;

struct State {
    next_instant: Instant,
    ramulator: RawRamulator,
}

struct Inner {
    // Completion callbacks run while Ramulator is ticked, so reads and writes
    // need independent counters that remain visible outside the model mutex.
    pending_reads: AtomicU32,
    pending_writes: AtomicU32,
    period: Duration,
    mutable: Mutex<State>,

    // Keep admission FIFO within each physical HBM channel.  A single global
    // lock causes head-of-line blocking: one full controller queue prevents
    // otherwise independent channels from accepting requests while it drains.
    // Ramulator has one controller per channel, so admission must be serialized
    // per channel rather than across the whole HBM system.
    read_locks: Vec<tokio::sync::Mutex<()>>,
    write_locks: Vec<tokio::sync::Mutex<()>>,
    admission_mapper: AdmissionMapper,

    // Size of a single transfer
    transfer_size: u32,
}

#[derive(Clone, Copy, Debug)]
enum AdmissionMapper {
    /// Conservative fallback for configurations whose mapper is not mirrored
    /// by this wrapper.
    Global,
    /// Ramulator2 HBM2 MOP4CLXOR, whose hierarchy before row is
    /// channel/pseudochannel/bankgroup/bank = C/1/2/2 bits.
    Hbm2Mop4Clxor { channels: u32 },
}

impl AdmissionMapper {
    fn lanes(self) -> usize {
        match self {
            Self::Global => 1,
            Self::Hbm2Mop4Clxor { channels } => channels as usize,
        }
    }

    fn lane(self, address: u64, transfer_size: u32) -> usize {
        match self {
            Self::Global => 0,
            Self::Hbm2Mop4Clxor { channels } => {
                debug_assert!(channels.is_power_of_two());
                debug_assert!(transfer_size.is_power_of_two());
                let channel_bits = channels.trailing_zeros();
                let channel_mask = u64::from(channels - 1);
                let mut value = address >> transfer_size.trailing_zeros();

                let column_low = value & 0b11;
                value >>= 2;
                let mut channel = value & channel_mask;
                value >>= channel_bits;
                // HBM2_8Gb: pseudochannel=1 bit, bankgroup=2, bank=2.
                value >>= 1 + 2 + 2;
                let column = column_low | ((value & 0b111) << 2);
                channel ^= column & channel_mask;
                channel as usize
            }
        }
    }
}

/// A wrapped ramulator that works with the event-based simulation.
#[derive(Clone)]
pub struct Ramulator(Arc<Inner>);

impl Ramulator {
    pub fn new(config: crate::config::Config) -> Result<Self> {
        let uses_hbm2_mop4clxor = config.dram["impl"].as_str() == Some("HBM2")
            && config.addr_mapper["impl"].as_str() == Some("MOP4CLXOR");
        let mut ramulator = RawRamulator::new(config)?;
        let period = Duration::from_picos(ramulator.period() as _);
        let transfer_size = ramulator.burst_size() * (ramulator.channel_width() / 8);
        let admission_mapper = if uses_hbm2_mop4clxor {
            let channels = ramulator.num_channels();
            if channels.is_power_of_two() {
                AdmissionMapper::Hbm2Mop4Clxor { channels }
            } else {
                AdmissionMapper::Global
            }
        } else {
            AdmissionMapper::Global
        };
        let admission_lanes = admission_mapper.lanes();

        Ok(Self(Arc::new(Inner {
            pending_reads: AtomicU32::new(0),
            pending_writes: AtomicU32::new(0),
            period,
            mutable: Mutex::new(State {
                // A request accepted at t=0 is first observed on the next
                // DRAM edge, not by a zero-time tick. This keeps event-runtime
                // completion cycles identical to standalone Ramulator ticks.
                next_instant: Instant::INIT + period,
                ramulator,
            }),

            read_locks: (0..admission_lanes)
                .map(|_| tokio::sync::Mutex::new(()))
                .collect(),
            write_locks: (0..admission_lanes)
                .map(|_| tokio::sync::Mutex::new(()))
                .collect(),
            admission_mapper,
            transfer_size,
        })))
    }

    pub fn period(&self) -> Duration {
        let mut guard = self.0.mutable.lock().unwrap();
        Duration::from_picos(guard.ramulator.period().into())
    }

    pub fn transfer_size(&self) -> u32 {
        self.0.transfer_size
    }

    pub fn pending_reads(&self) -> u32 {
        self.0.pending_reads.load(Ordering::Relaxed)
    }

    pub fn pending_writes(&self) -> u32 {
        self.0.pending_writes.load(Ordering::Relaxed)
    }

    pub fn pending_transactions(&self) -> u32 {
        self.pending_reads() + self.pending_writes()
    }

    fn tick(arc: Arc<Inner>) {
        let mut guard = arc.mutable.lock().unwrap();
        guard.ramulator.tick();
        guard.next_instant += arc.period;

        if arc.pending_reads.load(Ordering::Relaxed) + arc.pending_writes.load(Ordering::Relaxed)
            != 0
        {
            let arc = arc.clone();
            Executor::current().schedule(guard.next_instant, async { Self::tick(arc) });
        }
    }

    fn try_transfer(&self, addr: u64, write: bool) -> Result<impl Future<Output = ()>, ()> {
        let (send, recv) = tokio::sync::oneshot::channel();

        {
            let mut guard = self.0.mutable.lock().unwrap();

            // For max efficiency, we do not cycle the model unless a memory access is requested.
            if self.pending_transactions() == 0 {
                let now = Executor::current().now();
                while guard.next_instant < now {
                    guard.ramulator.tick();
                    guard.next_instant += self.0.period;
                }
            }

            let arc = self.0.clone();
            let completion = move || {
                if write {
                    arc.pending_writes.fetch_sub(1, Ordering::Relaxed);
                } else {
                    arc.pending_reads.fetch_sub(1, Ordering::Relaxed);
                }
                let _ = send.send(());
            };
            let success = if write {
                guard.ramulator.write(addr, completion)
            } else {
                guard.ramulator.read(addr, completion)
            };

            if !success {
                return Err(());
            }

            let previous_total = self.pending_transactions();
            if write {
                self.0.pending_writes.fetch_add(1, Ordering::Relaxed);
            } else {
                self.0.pending_reads.fetch_add(1, Ordering::Relaxed);
            }
            if previous_total == 0 {
                let arc = self.0.clone();
                Executor::current().schedule(guard.next_instant, async { Self::tick(arc) });
            }
        }

        Ok(async move {
            recv.await
                .expect("Ramulator completion sender dropped before callback")
        })
    }

    /// Submit a read request and wait for DRAM completion, not just acceptance.
    pub async fn read_transfer(&self, addr: u64) {
        let lane = self.0.admission_mapper.lane(addr, self.0.transfer_size);
        let guard = self.0.read_locks[lane].lock().await;
        let mut fut = self.try_transfer(addr, false);
        while fut.is_err() {
            Executor::current().resolve_at(self.0.period).await;
            fut = self.try_transfer(addr, false);
        }
        drop(guard);
        fut.unwrap().await
    }

    /// Submit a write request and wait for the Ramulator completion callback.
    pub async fn write_transfer(&self, addr: u64) {
        let lane = self.0.admission_mapper.lane(addr, self.0.transfer_size);
        let guard = self.0.write_locks[lane].lock().await;
        let mut fut = self.try_transfer(addr, true);
        while fut.is_err() {
            Executor::current().resolve_at(self.0.period).await;
            fut = self.try_transfer(addr, true);
        }
        drop(guard);
        fut.unwrap().await
    }

    /// Wait until every accepted read and write has reached completion.
    pub async fn drain(&self) {
        while self.pending_transactions() != 0 {
            Executor::current().resolve_at(self.0.period).await;
        }
    }
}

impl memory::MemoryTimingModel for Ramulator {
    async fn read(&self, addr: u64) {
        futures::future::join_all(
            (0..64)
                .step_by(self.0.transfer_size as _)
                .map(|offset| self.read_transfer(addr + offset)),
        )
        .await;
    }

    async fn write(&self, addr: u64) {
        futures::future::join_all(
            (0..64)
                .step_by(self.0.transfer_size as _)
                .map(|offset| self.write_transfer(addr + offset)),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn hbm2_config() -> crate::config::Config {
        crate::config::Config {
            dram: serde_json::json!({
                "impl": "HBM2",
                "org": { "preset": "HBM2_8Gb", "channel": 1 },
                "timing": { "preset": "HBM2_2Gbps" },
            }),
            controller: serde_json::json!({
                "impl": "Generic",
                "Scheduler": { "impl": "FRFCFS" },
                "RefreshManager": { "impl": "AllBank" },
                "RowPolicy": { "impl": "OpenRowPolicy" },
            }),
            addr_mapper: serde_json::json!({ "impl": "MOP4CLXOR" }),
        }
    }

    #[test]
    fn hbm2_mop4clxor_admission_lane_matches_mapper_bit_layout() {
        let mapper = AdmissionMapper::Hbm2Mop4Clxor { channels: 128 };
        assert_eq!(mapper.lane(0, 16), 0);
        assert_eq!(mapper.lane(16, 16), 1);
        assert_eq!(mapper.lane(32, 16), 2);
        assert_eq!(mapper.lane(48, 16), 3);
        assert_eq!(mapper.lane(64, 16), 1);
    }

    #[tokio::test]
    async fn read_and_write_resolve_on_completion_and_drain() {
        let ramulator = Ramulator::hbm2_preset(1).expect("HBM2 preset");
        let observer = ramulator.clone();
        let executor = Executor::new();

        executor.spawn(async move {
            let start = Executor::current().now();
            ramulator.read_transfer(0).await;
            let read_done = Executor::current().now();
            assert!(read_done > start);
            assert_eq!(ramulator.pending_reads(), 0);

            ramulator.write_transfer(64).await;
            let write_done = Executor::current().now();
            assert!(write_done > read_done);
            assert_eq!(ramulator.pending_writes(), 0);

            ramulator.drain().await;
            assert_eq!(ramulator.pending_transactions(), 0);
        });

        executor.enter(Instant::ETERNITY).await;
        assert_eq!(observer.pending_transactions(), 0);
    }

    #[tokio::test]
    async fn event_wrapper_matches_raw_read_completion_cycle() {
        let mut raw = RawRamulator::new(hbm2_config()).expect("raw HBM2 model");
        let raw_done = Arc::new(AtomicBool::new(false));
        let raw_flag = raw_done.clone();
        assert!(raw.read(0, move || {
            raw_flag.store(true, Ordering::Release);
        }));
        let raw_cycles = (1..=100_000)
            .find(|_| {
                raw.tick();
                raw_done.load(Ordering::Acquire)
            })
            .expect("raw read completion");

        let wrapped = Ramulator::new(hbm2_config()).expect("wrapped HBM2 model");
        let period_picos = wrapped.period().as_picos();
        let elapsed_picos = Arc::new(Mutex::new(None));
        let elapsed_out = elapsed_picos.clone();
        let executor = Executor::new();
        executor.spawn(async move {
            let start = Executor::current().now();
            wrapped.read_transfer(0).await;
            *elapsed_out.lock().unwrap() = Some((Executor::current().now() - start).as_picos());
        });
        executor.enter(Instant::ETERNITY).await;

        let wrapper_cycles =
            elapsed_picos.lock().unwrap().expect("wrapped completion") / period_picos;
        assert_eq!(wrapper_cycles, raw_cycles);
    }
}
