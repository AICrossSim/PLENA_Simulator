use core::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use runtime::{Duration, Executor, Instant};

use crate::raw::Ramulator as RawRamulator;

struct State {
    next_instant: Instant,
    ramulator: RawRamulator,
}

struct Inner {
    // Use atomic (but only use relaxed ordering) as this can be accessed while holding the mutex.
    pending_reads: AtomicU32,
    period: Duration,
    mutable: Mutex<State>,

    // Per-channel command-issue ports (FIFO; tokio mutex guarantees order).
    //
    // Hardware correspondence: each HBM (pseudo-)channel has its own
    // command/address bus and controller queue, so command issue scales
    // with channel count. The previous single global pair of locks
    // modelled ONE shared command port for the whole stack: when a
    // channel's queue filled, the head waiter spun one retry per DRAM
    // cycle while HOLDING the global lock, head-of-line-blocking every
    // other channel — capping aggregate issue at ~1 transfer/cycle and
    // flooring effective bandwidth at single-GB/s regardless of nominal
    // channel count (the "0.1% HBM utilization" pathology).
    //
    // Requests are striped over ports at `transfer_size` granularity.
    // This is an ISSUE-PORT model, not a replica of ramulator's internal
    // MOP4CLXOR address mapping: a mismatch only means two of our ports
    // occasionally contend for the same internal queue (each then spins
    // on its own port without blocking others) — backpressure semantics
    // survive, precision of which port stalls does not. Acceptable: the
    // DRAM model itself remains the bandwidth limiter, which is the
    // point.
    read_ports: Vec<tokio::sync::Mutex<()>>,
    write_ports: Vec<tokio::sync::Mutex<()>>,

    // Size of a single transfer
    transfer_size: u32,
}

/// A wrapped ramulator that works with the event-based simulation.
pub struct Ramulator(Arc<Inner>);

impl Ramulator {
    pub fn new(config: crate::config::Config) -> Result<Self> {
        let mut ramulator = RawRamulator::new(config)?;
        let period = Duration::from_picos(ramulator.period() as _);
        let transfer_size = ramulator.burst_size() * (ramulator.channel_width() / 8);
        let num_channels = ramulator.num_channels().max(1) as usize;

        Ok(Self(Arc::new(Inner {
            pending_reads: AtomicU32::new(0),
            period,
            mutable: Mutex::new(State {
                next_instant: Instant::INIT,
                ramulator,
            }),

            read_ports: (0..num_channels).map(|_| tokio::sync::Mutex::new(())).collect(),
            write_ports: (0..num_channels).map(|_| tokio::sync::Mutex::new(())).collect(),
            transfer_size,
        })))
    }

    /// Map a transfer address to a command-issue port. Striped at
    /// `transfer_size` granularity so consecutive transfers of one 64B
    /// line fan out across ports (channel-interleaved layout).
    fn port_of(&self, addr: u64) -> usize {
        (addr / self.0.transfer_size as u64) as usize % self.0.read_ports.len()
    }

    pub fn period(&self) -> Duration {
        let mut guard = self.0.mutable.lock().unwrap();
        Duration::from_picos(guard.ramulator.period().into())
    }

    pub fn transfer_size(&self) -> u32 {
        self.0.transfer_size
    }

    fn tick(arc: Arc<Inner>) {
        let mut guard = arc.mutable.lock().unwrap();
        guard.ramulator.tick();
        guard.next_instant += arc.period;

        if arc
            .pending_reads
            .load(core::sync::atomic::Ordering::Relaxed)
            != 0
        {
            let arc = arc.clone();
            Executor::current().schedule(guard.next_instant, async { Self::tick(arc) });
        }
    }

    /// Send a read request to ramulator.
    fn try_read(&self, addr: u64) -> Result<impl Future<Output = ()>, ()> {
        let (send, recv) = tokio::sync::oneshot::channel();

        {
            let mut guard = self.0.mutable.lock().unwrap();

            // For max efficiency, we do not cycle the model unless a memory access is requested.
            if self
                .0
                .pending_reads
                .load(core::sync::atomic::Ordering::Relaxed)
                == 0
            {
                let now = Executor::current().now();
                while guard.next_instant < now {
                    guard.ramulator.tick();
                    guard.next_instant += self.0.period;
                }
            }

            let arc = self.0.clone();
            let success = guard.ramulator.read(addr, move || {
                arc.pending_reads
                    .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
                let _ = send.send(());
            });

            if !success {
                return Err(());
            }

            if self
                .0
                .pending_reads
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
                == 0
            {
                let arc = self.0.clone();
                Executor::current().schedule(guard.next_instant, async { Self::tick(arc) });
            }
        }

        Ok(async { recv.await.unwrap() })
    }

    /// Send a read request to ramulator via the address's issue port.
    /// The port lock is held only until ramulator ACCEPTS the request
    /// (retrying once per DRAM cycle on a full queue — per-channel
    /// backpressure), then dropped before awaiting completion, so each
    /// port can pipeline many in-flight requests.
    pub async fn read_transfer(&self, addr: u64) {
        let guard = self.0.read_ports[self.port_of(addr)].lock().await;
        let mut fut = self.try_read(addr);
        while fut.is_err() {
            Executor::current().resolve_at(self.0.period).await;
            fut = self.try_read(addr);
        }
        drop(guard);
        fut.unwrap().await
    }

    /// Send a write request to ramulator.
    fn try_write_transfer(&self, addr: u64) -> Result<(), ()> {
        let mut guard = self.0.mutable.lock().unwrap();

        // For max efficiency, we do not cycle the model unless a memory access is requested.
        if self
            .0
            .pending_reads
            .load(core::sync::atomic::Ordering::Relaxed)
            == 0
        {
            let now = Executor::current().now();
            while guard.next_instant < now {
                guard.ramulator.tick();
                guard.next_instant += self.0.period;
            }
        }

        guard.ramulator.write(addr).then_some(()).ok_or(())
    }

    /// Send a write request to ramulator via the address's issue port.
    pub async fn write_transfer(&self, addr: u64) {
        let _guard = self.0.write_ports[self.port_of(addr)].lock().await;
        while self.try_write_transfer(addr).is_err() {
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
