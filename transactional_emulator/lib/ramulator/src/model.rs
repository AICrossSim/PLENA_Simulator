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
    pending_accesses: AtomicU32,
    period: Duration,
    mutable: Mutex<State>,

    // A queue for requests that Ramulator failed to accept. Note that tokio mutex guarantees FIFO order.
    lock: tokio::sync::Mutex<()>,

    // Size of a single transfer
    transfer_size: u32,
}

/// Investigation-only counters. Each `schedule(tick)` issued from `try_access`
/// begins an independent self-rescheduling tick chain; a chain ends when a tick
/// sees no outstanding accesses and declines to reschedule. At most one chain
/// should ever be live.
pub mod probe {
    use core::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};

    pub static TICKS: AtomicU64 = AtomicU64::new(0);
    pub static LIVE_CHAINS: AtomicI64 = AtomicI64::new(0);
    pub static MAX_LIVE_CHAINS: AtomicI64 = AtomicI64::new(0);
    pub static CHAIN_STARTS: AtomicU64 = AtomicU64::new(0);

    pub fn chain_started() {
        CHAIN_STARTS.fetch_add(1, Relaxed);
        let live = LIVE_CHAINS.fetch_add(1, Relaxed) + 1;
        MAX_LIVE_CHAINS.fetch_max(live, Relaxed);
    }

    pub fn chain_ended() {
        LIVE_CHAINS.fetch_sub(1, Relaxed);
    }

    pub fn reset() {
        TICKS.store(0, Relaxed);
        LIVE_CHAINS.store(0, Relaxed);
        MAX_LIVE_CHAINS.store(0, Relaxed);
        CHAIN_STARTS.store(0, Relaxed);
    }
}

/// A wrapped ramulator that works with the event-based simulation.
pub struct Ramulator(Arc<Inner>);

impl Ramulator {
    pub fn new(config: crate::config::Config) -> Result<Self> {
        let mut ramulator = RawRamulator::new(config)?;
        let period = Duration::from_picos(ramulator.period() as _);
        let transfer_size = ramulator.burst_size() * (ramulator.channel_width() / 8);

        Ok(Self(Arc::new(Inner {
            pending_accesses: AtomicU32::new(0),
            period,
            mutable: Mutex::new(State {
                next_instant: Instant::INIT,
                ramulator,
            }),

            lock: tokio::sync::Mutex::new(()),
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

    /// The instant the DRAM model itself has been advanced to.
    ///
    /// Invariant: this must never run ahead of `Executor::now()`. The catch-up
    /// loop in `try_access` can only repair lag, so any lead is permanent.
    pub fn dram_clock(&self) -> Instant {
        self.0.mutable.lock().unwrap().next_instant
    }

    /// Outstanding accesses, as the ticker-start guard sees them.
    pub fn pending_accesses(&self) -> u32 {
        self.0
            .pending_accesses
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    fn tick(arc: Arc<Inner>) {
        let mut guard = arc.mutable.lock().unwrap();
        guard.ramulator.tick();
        guard.next_instant += arc.period;
        probe::TICKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        if arc
            .pending_accesses
            .load(core::sync::atomic::Ordering::Relaxed)
            != 0
        {
            let arc = arc.clone();
            Executor::current().schedule(guard.next_instant, async { Self::tick(arc) });
        } else {
            probe::chain_ended();
        }
    }

    /// Send a request to ramulator.
    fn try_access(&self, addr: u64, write: bool) -> Result<impl Future<Output = ()>, ()> {
        let (send, recv) = tokio::sync::oneshot::channel();

        {
            let mut guard = self.0.mutable.lock().unwrap();

            // For max efficiency, we do not cycle the model unless a memory access is requested.
            if self
                .0
                .pending_accesses
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
            let success = guard.ramulator.access(addr, write, move || {
                arc.pending_accesses
                    .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
                let _ = send.send(());
            });

            if !success {
                return Err(());
            }

            if self
                .0
                .pending_accesses
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
                == 0
            {
                probe::chain_started();
                let arc = self.0.clone();
                Executor::current().schedule(guard.next_instant, async { Self::tick(arc) });
            }
        }

        Ok(async { recv.await.unwrap() })
    }

    /// Send a request to ramulator.
    pub async fn access(&self, addr: u64, write: bool) {
        let guard = self.0.lock.lock().await;
        let mut fut = self.try_access(addr, write);
        while fut.is_err() {
            Executor::current().resolve_at(self.0.period).await;
            fut = self.try_access(addr, write);
        }
        drop(guard);
        fut.unwrap().await
    }

    /// Send a read request to ramulator.
    pub async fn read_transfer(&self, addr: u64) {
        self.access(addr, false).await
    }

    /// Send a write request to ramulator.
    pub async fn write_transfer(&self, addr: u64) {
        self.access(addr, true).await
    }
}

impl memory::MemoryTimingModel for Ramulator {
    async fn read(&self, addr: u64) {
        let transfers: Vec<_> = (0..64u64)
            .step_by(self.0.transfer_size as usize)
            .map(|offset| self.read_transfer(addr + offset))
            .collect();
        futures::future::join_all(transfers).await;
    }

    async fn write(&self, addr: u64) {
        let transfers: Vec<_> = (0..64u64)
            .step_by(self.0.transfer_size as usize)
            .map(|offset| self.write_transfer(addr + offset))
            .collect();
        futures::future::join_all(transfers).await;
    }
}

#[cfg(test)]
mod double_ticker_investigation {
    use super::*;

    fn report(label: &str, ex: &Executor, ram: &Ramulator) {
        use core::sync::atomic::Ordering::Relaxed;
        let now = ex.now() - Instant::INIT;
        let ticks = probe::TICKS.load(Relaxed);
        let ns = now.as_picos() / 1000;
        println!(
            "{label:<26} sim={:>5}ns  ticks={:>5}  ticks/ns={:.2}  chains_started={}  max_live_chains={}  dram_lead={}ns",
            ns,
            ticks,
            ticks as f64 / (ns as f64).max(1.0),
            probe::CHAIN_STARTS.load(Relaxed),
            probe::MAX_LIVE_CHAINS.load(Relaxed),
            (ram.dram_clock() - ex.now()).as_picos() as i64 / 1000,
        );
    }

    /// Baseline: concurrent writes to distinct addresses cannot coalesce.
    #[tokio::test]
    async fn distinct_addresses_keep_one_chain() {
        probe::reset();
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let ex = Executor::new();
        for i in 0..2u64 {
            let r = ram.clone();
            ex.spawn(async move { r.write_transfer(0x1000 + i * 4096).await });
        }
        ex.enter(Instant::INIT + Duration::from_micros(50)).await;
        report("distinct addresses", &ex, &ram);
        assert_eq!(
            probe::MAX_LIVE_CHAINS.load(core::sync::atomic::Ordering::Relaxed),
            1,
            "baseline should never have two tick chains live"
        );
    }

    /// Two concurrent writes to the SAME address. Ramulator's write-coalescing
    /// path fires the incoming request's callback synchronously inside
    /// `access()`, so `pending_accesses` dips 1 -> 0 before `try_access` reaches
    /// its `fetch_add(1) == 0` guard. The guard reads 0 and starts a second
    /// tick chain beside the live one.
    #[tokio::test]
    async fn same_address_writes_start_a_second_chain() {
        probe::reset();
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let ex = Executor::new();
        const ADDR: u64 = 0x4000;
        for _ in 0..2 {
            let r = ram.clone();
            ex.spawn(async move { r.write_transfer(ADDR).await });
        }
        ex.enter(Instant::INIT + Duration::from_micros(50)).await;
        report("same address", &ex, &ram);
        assert_eq!(
            probe::MAX_LIVE_CHAINS.load(core::sync::atomic::Ordering::Relaxed),
            1,
            "two tick chains ran concurrently -- the DRAM model is being ticked twice per cycle"
        );
    }

    /// Sustained: a coalescing write, then traffic that keeps the chains alive
    /// long enough for the doubled tick rate to be unmistakable.
    #[tokio::test]
    async fn doubled_chains_double_the_tick_rate() {
        probe::reset();
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let ex = Executor::new();
        const ADDR: u64 = 0x4000;
        for _ in 0..2 {
            let r = ram.clone();
            ex.spawn(async move { r.write_transfer(ADDR).await });
        }
        // Keep accesses outstanding so neither chain retires early.
        let r = ram.clone();
        ex.spawn(async move {
            for i in 0..400u64 {
                r.read_transfer(0x100000 + i * 64).await;
            }
        });
        ex.enter(Instant::INIT + Duration::from_micros(500)).await;
        report("same address, sustained", &ex, &ram);

        probe::reset();
        let ram2 = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let ex2 = Executor::new();
        for i in 0..2u64 {
            let r = ram2.clone();
            ex2.spawn(async move { r.write_transfer(0x1000 + i * 4096).await });
        }
        let r = ram2.clone();
        ex2.spawn(async move {
            for i in 0..400u64 {
                r.read_transfer(0x100000 + i * 64).await;
            }
        });
        ex2.enter(Instant::INIT + Duration::from_micros(500)).await;
        report("distinct addrs, sustained", &ex2, &ram2);
    }
}

#[cfg(test)]
mod skew_accumulation {
    use super::*;
    use core::sync::atomic::Ordering::Relaxed;

    /// Fire the coalescing race `rounds` times and report how far the DRAM clock
    /// has drifted ahead of simulated time. One period of lead is normal (the
    /// ticker pre-advances `next_instant` for its next tick); anything beyond
    /// that is skew the catch-up loop can never repair.
    async fn drift_after(rounds: u64, same_addr: bool) -> (i64, i64, u64) {
        probe::reset();
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let ex = Executor::new();

        for round in 0..rounds {
            let addr = 0x40000 + round * 64;
            let r1 = ram.clone();
            ex.spawn(async move { r1.write_transfer(addr).await });
            let r2 = ram.clone();
            let second = if same_addr { addr } else { addr + 0x100000 };
            ex.spawn(async move { r2.write_transfer(second).await });
            ex.enter(Instant::INIT + Duration::from_micros(10 * (round + 1)))
                .await;
        }
        ex.enter(Instant::ETERNITY).await;

        let lead = (ram.dram_clock() - ex.now()).as_picos() as i64 / 1000;
        (
            lead,
            probe::MAX_LIVE_CHAINS.load(Relaxed),
            probe::CHAIN_STARTS.load(Relaxed),
        )
    }

    #[tokio::test]
    async fn skew_grows_one_cycle_per_coalescing_event() {
        println!(
            "{:>8} | {:>18} | {:>18}",
            "rounds", "same-addr lead(ns)", "distinct lead(ns)"
        );
        println!("{}", "-".repeat(52));
        for rounds in [1u64, 2, 4, 8, 16, 32] {
            let (bad, bad_max, _) = drift_after(rounds, true).await;
            let (good, good_max, _) = drift_after(rounds, false).await;
            println!(
                "{rounds:>8} | {bad:>18} | {good:>18}    (max_live: same={bad_max} distinct={good_max})"
            );
        }
    }
}

#[cfg(test)]
mod chain_stress {
    use super::*;
    use core::sync::atomic::Ordering::Relaxed;

    /// Hammer the coalescing path with many writers to the same address while
    /// keeping `pending_accesses` oscillating around 1, to find the worst-case
    /// number of concurrently live tick chains.
    #[tokio::test]
    async fn how_many_chains_can_stack() {
        for writers in [2usize, 4, 8, 16, 64] {
            probe::reset();
            let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
            let ex = Executor::new();
            const ADDR: u64 = 0x4000;
            for _ in 0..writers {
                let r = ram.clone();
                ex.spawn(async move { r.write_transfer(ADDR).await });
            }
            ex.enter(Instant::INIT + Duration::from_micros(500)).await;
            println!(
                "writers={writers:<3} chains_started={:<4} max_live_chains={:<3} ticks={:<6} sim={}ns  dram_lead={}ns",
                probe::CHAIN_STARTS.load(Relaxed),
                probe::MAX_LIVE_CHAINS.load(Relaxed),
                probe::TICKS.load(Relaxed),
                (ex.now() - Instant::INIT).as_picos() / 1000,
                (ram.dram_clock() - ex.now()).as_picos() as i64 / 1000,
            );
        }
    }
}
