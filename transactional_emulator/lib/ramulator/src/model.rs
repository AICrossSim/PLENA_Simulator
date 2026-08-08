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
