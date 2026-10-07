use core::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use runtime::{Duration, Executor, Instant};

use crate::raw::Ramulator as RawRamulator;

struct State {
    next_instant: Instant,
    ramulator: RawRamulator,

    /// Whether a tick loop is currently driving the model.
    ///
    /// Exactly one may be live. This cannot be inferred from `pending_accesses`:
    /// the model completes some requests synchronously (see `try_access`), so
    /// that counter can dip to zero and back while a loop is still running.
    ticker_running: bool,
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
                ticker_running: false,
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

    /// Drive the model one cycle per period for as long as accesses are outstanding.
    ///
    /// This is a single task that awaits a fresh timer each cycle rather than a
    /// task per cycle: the busy stretches of a run are millions of cycles long,
    /// and spawning there costs an `Arc<Task>` plus a boxed future every time.
    /// Timer identities are still allocated at the same point in each cycle, so
    /// same-instant event ordering is unchanged.
    async fn tick_loop(arc: Arc<Inner>) {
        loop {
            let timer = {
                let mut guard = arc.mutable.lock().unwrap();
                guard.ramulator.tick();
                guard.next_instant += arc.period;

                if arc
                    .pending_accesses
                    .load(core::sync::atomic::Ordering::Relaxed)
                    == 0
                {
                    guard.ticker_running = false;
                    return;
                }

                Executor::current().resolve_at(guard.next_instant)
            };
            timer.await;
        }
    }

    /// Start the ticking task, with the first cycle due at `due`.
    ///
    /// Mirrors what `Executor::schedule` does internally, so the leading timer
    /// is created at the same moment it was before.
    fn start_ticking(arc: Arc<Inner>, due: Instant) {
        let executor = Executor::current();
        let timer = executor.resolve_at(due);
        executor.spawn(async move {
            timer.await;
            Self::tick_loop(arc).await;
        });
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

            // Count the access before handing it over: the model may complete it
            // *synchronously* from inside `access` -- a write to an address it is
            // already buffering is absorbed on the spot and the callback runs
            // before this call returns. Incrementing first keeps the callback's
            // decrement from underflowing, and keeps the count below honest.
            self.0
                .pending_accesses
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

            let arc = self.0.clone();
            let success = guard.ramulator.access(addr, write, move || {
                arc.pending_accesses
                    .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
                let _ = send.send(());
            });

            if !success {
                // Rejected: the callback was dropped without running, so undo.
                self.0
                    .pending_accesses
                    .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
                return Err(());
            }

            // Drive the model only if something is still outstanding, and only if
            // nothing is driving it already. Testing `pending_accesses` alone would
            // start a second loop whenever a synchronous completion dropped the
            // count to zero underneath a live one.
            if !guard.ticker_running
                && self
                    .0
                    .pending_accesses
                    .load(core::sync::atomic::Ordering::Relaxed)
                    != 0
            {
                guard.ticker_running = true;
                Self::start_ticking(self.0.clone(), guard.next_instant);
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
impl Ramulator {
    /// The instant the DRAM model itself has been advanced to.
    fn dram_clock(&self) -> Instant {
        self.0.mutable.lock().unwrap().next_instant
    }

    fn ticker_running(&self) -> bool {
        self.0.mutable.lock().unwrap().ticker_running
    }
}

#[cfg(test)]
mod ticker_tests {
    use super::*;

    /// The model is driven one cycle at a time by a single chain of scheduled
    /// ticks, so its clock sits at most one period past simulated time -- that
    /// one period is the tick already queued. Any larger lead means it was
    /// ticked more often than time passed, and `try_access`'s catch-up loop
    /// (`while next_instant < now`) can only repair lag, never lead.
    fn assert_clock_sane(label: &str, ex: &Executor, ram: &Ramulator, period: Duration) {
        let lead = ram.dram_clock() - ex.now();
        assert!(
            lead <= period,
            "{label}: DRAM clock leads simulated time by {lead:?} (at most {period:?} expected) \
             -- the model was over-ticked"
        );
    }

    /// Writes to an address the model is already buffering are absorbed on the
    /// spot, completing synchronously inside `access`. Before the fix that made
    /// `pending_accesses` dip to zero under a live tick chain, so the guard
    /// started another one -- once per concurrent writer.
    #[tokio::test]
    async fn concurrent_writes_to_one_address_keep_a_single_ticker() {
        const ADDR: u64 = 0x4000;
        for writers in [2usize, 4, 8, 16, 64] {
            let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
            let period = ram.0.period;
            let ex = Executor::new();
            for _ in 0..writers {
                let r = ram.clone();
                ex.spawn(async move { r.write_transfer(ADDR).await });
            }
            ex.enter(Instant::INIT + Duration::from_micros(500)).await;

            assert_clock_sane(
                &format!("{writers} same-address writers"),
                &ex,
                &ram,
                period,
            );
            assert!(
                !ram.ticker_running(),
                "{writers} writers: a tick loop outlived the last access"
            );
        }
    }

    /// Writes to distinct addresses cannot coalesce; this is the control.
    #[tokio::test]
    async fn concurrent_writes_to_distinct_addresses_keep_a_single_ticker() {
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let period = ram.0.period;
        let ex = Executor::new();
        for i in 0..64u64 {
            let r = ram.clone();
            ex.spawn(async move { r.write_transfer(0x4000 + i * 4096).await });
        }
        ex.enter(Instant::INIT + Duration::from_micros(500)).await;
        assert_clock_sane("64 distinct-address writers", &ex, &ram, period);
    }

    /// Saturating the model's buffers makes `access` reject, which drops the
    /// callback without running it -- so the submission-side increment has to be
    /// undone by hand. If that undo is wrong the counter never returns to zero
    /// and the tick chain runs forever.
    #[tokio::test]
    async fn rejected_accesses_do_not_leak_the_outstanding_count() {
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let period = ram.0.period;
        let ex = Executor::new();

        // Far more concurrent writes than any single controller will buffer, so a
        // large share of the submissions are refused and retried.
        let done = Arc::new(AtomicU32::new(0));
        for i in 0..512u64 {
            let r = ram.clone();
            let d = done.clone();
            ex.spawn(async move {
                r.write_transfer(0x20000 + i * 64).await;
                d.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            });
        }
        // Bounded deliberately: a leaked count keeps the tick chain alive forever,
        // and `Instant::ETERNITY` would turn that into a hung test rather than a
        // failing one.
        ex.enter(Instant::INIT + Duration::from_micros(500)).await;

        assert_eq!(
            done.load(core::sync::atomic::Ordering::Relaxed),
            512,
            "not every write completed within the deadline"
        );

        assert_eq!(
            ram.0
                .pending_accesses
                .load(core::sync::atomic::Ordering::Relaxed),
            0,
            "outstanding count did not return to zero after all accesses completed"
        );
        assert!(
            !ram.ticker_running(),
            "a tick loop outlived the last access"
        );
        assert_clock_sane("512 writers with rejections", &ex, &ram, period);
    }

    /// Reads against an address the model is already buffering a write for take
    /// its read-forwarding path -- the one place a read could plausibly acquire
    /// the same inline-callback hazard writes have. It does not: forwarding
    /// defers through `m_pending` with `depart = m_clk + 1`. The concurrent write
    /// is what puts the address in the buffered set; without it the forwarding
    /// branch is unreachable and this only exercises the ordinary read queue.
    #[tokio::test]
    async fn reads_forwarded_from_a_buffered_write_keep_a_single_ticker() {
        const ADDR: u64 = 0xC000;
        let ram = Arc::new(Ramulator::hbm2_preset(1).unwrap());
        let period = ram.0.period;
        let ex = Executor::new();

        let w = ram.clone();
        ex.spawn(async move { w.write_transfer(ADDR).await });
        for _ in 0..64 {
            let r = ram.clone();
            ex.spawn(async move { r.read_transfer(ADDR).await });
        }

        ex.enter(Instant::INIT + Duration::from_micros(500)).await;
        assert_clock_sane(
            "64 readers forwarded from a buffered write",
            &ex,
            &ram,
            period,
        );
        assert!(
            !ram.ticker_running(),
            "a tick loop outlived the last access"
        );
    }
}
