use core::pin::Pin;
use std::cell::{Cell, UnsafeCell};
use std::collections::{BTreeSet, VecDeque};
use std::marker::PhantomPinned;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Wake, Waker};

use crate::{Deadline, Instant};

/// Monotonic, process-global timer-creation counter. Used as the
/// secondary key in `Timer`'s `Ord` so two timers with identical
/// `resolve_at` are woken in the order they were registered, instead
/// of in allocator-pointer order. The OOO dispatcher (`yw/ooo_arch`)
/// turns one op into a spawned task with its own `cycle!()` chain,
/// dramatically increasing the number of simultaneous-instant timers
/// — and pointer order is allocator-dependent (e.g. an alloc-freed
/// hot slot can have a smaller address than a newly-grabbed one), so
/// the same kernel would produce slightly different cycle-event
/// orderings across runs. Insertion order is stable across processes,
/// machines, and rebuilds.
static TIMER_SEQ: AtomicU64 = AtomicU64::new(0);

pub struct ResolveAt(Timer);

impl Future for ResolveAt {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe { self.map_unchecked_mut(|x| &mut x.0) }.poll(cx)
    }
}

struct Timer {
    executor: Executor,
    resolve_at: Instant,
    /// Insertion-order tie-breaker for same-`resolve_at` siblings.
    /// See [`TIMER_SEQ`] for rationale. Unique per process lifetime.
    seq: u64,
    waker: Option<Waker>,
    _phantom: PhantomPinned,
}

impl PartialEq for Timer {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self, other)
    }
}

impl Eq for Timer {}

impl Ord for Timer {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Primary: simulated time. Secondary: insertion order (stable
        // across processes/machines — see TIMER_SEQ above). Identity
        // (same `self`) is the implicit final differentiator since
        // `seq` is unique per Timer, so the BTreeSet can hold multiple
        // distinct entries even when `resolve_at` matches.
        self.resolve_at
            .cmp(&other.resolve_at)
            .then(self.seq.cmp(&other.seq))
    }
}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let mut executor = self.executor.0.lock().unwrap();
        executor.timers.remove(self);
    }
}

impl Future for Timer {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let lifetime_extended: &'static mut Timer = unsafe { std::mem::transmute(&mut *this) };

        let mut executor = this.executor.0.lock().unwrap();

        if executor.now >= this.resolve_at {
            // To prevent mistaken elapsed into the past.
            assert_eq!(executor.now, this.resolve_at);
            return Poll::Ready(());
        }

        // Not yet registered, go register in the executor.
        if this.waker.is_none() {
            executor.timers.insert(lifetime_extended);
        }

        this.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// A timing simulator task.
struct Task {
    executor: Weak<Mutex<ExecutorInner>>,

    // May only be accessed by the `executor`.
    task: UnsafeCell<Option<Box<dyn Future<Output = ()> + Send>>>,
}

unsafe impl Send for Task {}
unsafe impl Sync for Task {}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self, other)
    }
}

impl Wake for Task {
    fn wake(self: Arc<Self>) {
        let Some(executor) = self.executor.upgrade() else {
            return;
        };
        let mut executor = executor.lock().unwrap();
        executor.ready_tasks.push_back(self.clone());
    }
}

struct ExecutorInner {
    now: Instant,
    ready_tasks: VecDeque<Arc<Task>>,
    timers: BTreeSet<&'static mut Timer>,
}

/// An executor for timing simulations.
#[derive(Clone)]
pub struct Executor(Arc<Mutex<ExecutorInner>>);

thread_local! {
    static EXECUTOR: Cell<Option<Executor>> = Cell::new(None);
}

impl Executor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(ExecutorInner {
            now: Instant::INIT,
            ready_tasks: VecDeque::new(),
            timers: BTreeSet::new(),
        })))
    }

    /// Obtain the associated executor with current thread.
    pub fn current() -> Self {
        EXECUTOR.with(|x| {
            let v = x.take();
            x.set(v.clone());
            v.unwrap()
        })
    }

    /// Return the simulation instant this executor is at.
    pub fn now(&self) -> Instant {
        // TODO: Optimize this to be lock-free.
        self.0.lock().unwrap().now
    }

    /// Create a timer that is resolved at later time.
    pub fn resolve_at(&self, fire_at: impl Deadline) -> ResolveAt {
        let instant = fire_at.to_instant(self.now());
        ResolveAt(Timer {
            executor: self.clone(),
            resolve_at: instant,
            // Relaxed is sufficient: we only need monotonicity per
            // observer (each Timer reads `seq` once at construction
            // and uses it for Ord, no inter-thread synchronization).
            seq: TIMER_SEQ.fetch_add(1, AtomicOrdering::Relaxed),
            waker: None,
            _phantom: PhantomPinned,
        })
    }

    /// Schedule an event to be executed at future time.
    pub fn schedule(
        &self,
        fire_at: impl Deadline,
        task: impl Future<Output = ()> + Send + 'static,
    ) {
        let fut = self.resolve_at(fire_at);
        self.spawn(async move {
            fut.await;
            task.await
        });
    }

    /// Run a task at current instant.
    pub fn spawn(&self, task: impl Future<Output = ()> + Send + 'static) {
        let mut inner = self.0.lock().unwrap();
        let task = Arc::new(Task {
            executor: Arc::downgrade(&self.0),
            task: UnsafeCell::new(Some(Box::new(task))),
        });
        inner.ready_tasks.push_back(task);
    }

    /// Start running simulation.
    ///
    /// The future will be completed once no more events are scheduled, or when the set instant is reached.
    pub async fn enter(&self, timeout: Instant) {
        let mut guard = self.0.lock().unwrap();

        // `coop::unconstrained` below disables tokio's per-poll budget so we
        // never get pre-empted in the middle of polling a sim task. That's
        // necessary for correctness but it also means a long enter() call
        // would never yield back to the surrounding tokio runtime, which
        // starves any concurrent tokio task (e.g. a progress poll arriving
        // over the online service). We yield manually every Nth task poll
        // — frequent enough that ~kHz GUI polls get through during a heavy
        // kernel, sparse enough that the per-poll yield cost is amortized
        // over real work.
        const TOKIO_YIELD_EVERY: u32 = 256;
        let mut polled_since_yield: u32 = 0;

        loop {
            // First, continue polling ready tasks until there aren't any.
            while let Some(task) = guard.ready_tasks.pop_front() {
                let waker = task.clone().into();
                let mut cx = Context::from_waker(&waker);

                drop(guard);

                if let Some(fut) = unsafe { (*task.task.get()).as_deref_mut() } {
                    let poll = tokio::task::coop::unconstrained(std::future::poll_fn(|_| {
                        EXECUTOR.with(|x| {
                            assert!(x.replace(Some(self.clone())).is_none());
                            let ret = unsafe { Pin::new_unchecked(&mut *fut) }.poll(&mut cx);
                            x.set(None);
                            Poll::Ready(ret)
                        })
                    }))
                    .await;

                    // Drop the future once it's ready so it's not polled again.
                    if poll.is_ready() {
                        unsafe {
                            *task.task.get() = None;
                        }
                    }
                }

                polled_since_yield = polled_since_yield.wrapping_add(1);
                if polled_since_yield >= TOKIO_YIELD_EVERY {
                    polled_since_yield = 0;
                    tokio::task::yield_now().await;
                }

                guard = self.0.lock().unwrap();
            }

            let Some(timer) = guard.timers.pop_first() else {
                drop(guard);
                // No more timers -- completes!
                return;
            };

            // Timeouts.
            if timer.resolve_at >= timeout {
                drop(guard);
                return;
            }

            // Ensure time never go backwards.
            assert!(timer.resolve_at >= guard.now);

            // Progress current simulated time to the next timer.
            // This should be alright, as we're outside any polled tasks, so `Executor::current()` may no longer be accessed by polled tasks,
            guard.now = timer.resolve_at;

            drop(guard);
            timer.waker.as_ref().unwrap().wake_by_ref();
            guard = self.0.lock().unwrap();

            // Trigger all timers of the same instant together.
            while let Some(timer) = guard.timers.first()
                && timer.resolve_at <= guard.now
            {
                assert_eq!(timer.resolve_at, guard.now);
                let timer = guard.timers.pop_first().unwrap();
                drop(guard);
                timer.waker.as_ref().unwrap().wake_by_ref();
                guard = self.0.lock().unwrap();
            }
        }
    }
}
