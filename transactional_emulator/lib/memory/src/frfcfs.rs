use std::marker::PhantomPinned;
use std::mem::forget;
use std::pin::{Pin, pin};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures_util::task::AtomicWaker;
use runtime::{Duration, Executor};

pub struct FrFcFs {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    precharge: Duration,
    rcd: Duration,
    // These are of shorter lifetime, but pinned so cannot go way while in this vector.
    req: Vec<&'static AcquireFuture>,
    idle: bool,
}

/// Permit to operate on the row.
pub struct Permit {
    row: u32,
    inner: Arc<Mutex<Inner>>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();

        if inner.req.is_empty() {
            let inner_clone = self.inner.clone();

            // Close the row returning back to idle if no more requests.
            Executor::current().schedule(inner.precharge, async move {
                let mut inner = inner_clone.lock().unwrap();

                if inner.req.is_empty() {
                    inner.idle = true;
                    return;
                }

                let req = inner.req.remove(0);
                req.completed.store(true, Ordering::Relaxed);
                req.waker.wake();
            });

            return;
        }

        // Find the next request in the same row.
        for (i, req) in inner.req.iter().enumerate() {
            if req.row == self.row {
                let req = inner.req.remove(i);
                req.completed.store(true, Ordering::Relaxed);
                req.waker.wake();
                return;
            }
        }

        let req = inner.req.remove(0);

        // If nothing find, take the first.
        Executor::current().schedule(inner.precharge + inner.rcd, async {
            req.completed.store(true, Ordering::Relaxed);
            req.waker.wake();
        });
    }
}

// I am too lazy to implement proper cancellation logic..
// Just make it not cancellable.
struct CancellationGuard(PhantomPinned);

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        extern "C" fn blow_up() {
            panic!("this future cannot be cancelled");
        }
        blow_up();
    }
}

struct AcquireFuture {
    inner: Arc<Mutex<Inner>>,
    row: u32,
    completed: AtomicBool,
    waker: AtomicWaker,
}

impl Future for AcquireFuture {
    type Output = Permit;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.waker.register(cx.waker());

        if self.completed.load(Ordering::Relaxed) {
            return Poll::Ready(Permit {
                row: self.row,
                inner: self.inner.clone(),
            });
        }

        Poll::Pending
    }
}

impl FrFcFs {
    pub fn new(tck: Duration, rcd: u32, rp: u32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                precharge: tck * rp,
                rcd: tck * rcd,
                req: Vec::new(),
                idle: true,
            })),
        }
    }

    /// Simulate row activation.
    pub async fn acquire(&self, row: u32) -> Permit {
        // The algorithm isn't written with cancellation in mind.
        // Doing this will cause the future to be pinned, and will blow up the process
        // if misused.
        let guard = CancellationGuard(PhantomPinned);

        // Pin this in place so that we don't move it by mistake.
        let fut = pin!(AcquireFuture {
            inner: self.inner.clone(),
            row,
            completed: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        });

        {
            // SAFETY: we pinned this future.
            let fut_ref = unsafe { &*&raw const *fut };
            let mut inner = self.inner.lock().unwrap();

            if inner.idle {
                inner.idle = false;
                Executor::current().schedule(inner.rcd, async {
                    fut_ref.completed.store(true, Ordering::Relaxed);
                    fut_ref.waker.wake();
                });
            } else {
                inner.req.push(fut_ref);
            }
        }

        let permit = fut.await;
        forget(guard);
        permit
    }
}
