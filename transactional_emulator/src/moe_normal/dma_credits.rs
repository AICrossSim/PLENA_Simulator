//! Borrowable per-core minimum credits. Reservations apply only while another
//! core is waiting, so an idle core cannot strand shared response storage.
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

struct State {
    used: Vec<usize>,
    waiting: Vec<usize>,
}
pub(super) struct CreditPool {
    total: usize,
    reserve: usize,
    state: Mutex<State>,
    changed: Notify,
}

pub(super) struct Credit {
    pool: Arc<CreditPool>,
    core: usize,
}
struct Waiting {
    pool: Arc<CreditPool>,
    core: usize,
    active: bool,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.active {
            self.pool.state.lock().unwrap().waiting[self.core] -= 1;
            self.pool.changed.notify_waiters();
        }
    }
}
impl Drop for Credit {
    fn drop(&mut self) {
        self.pool.state.lock().unwrap().used[self.core] -= 1;
        self.pool.changed.notify_waiters();
    }
}
impl CreditPool {
    pub(super) fn new(total: usize, cores: usize) -> Self {
        assert!(cores > 0 && total >= cores);
        Self {
            total,
            reserve: total / 8,
            state: Mutex::new(State {
                used: vec![0; cores],
                waiting: vec![0; cores],
            }),
            changed: Notify::new(),
        }
    }
    pub(super) async fn acquire(self: &Arc<Self>, core: usize) -> Credit {
        self.state.lock().unwrap().waiting[core] += 1;
        let mut waiting = Waiting {
            pool: self.clone(),
            core,
            active: true,
        };
        loop {
            let notification = self.changed.notified();
            tokio::pin!(notification);
            // Register before checking state; a release between the check and
            // await must not be lost. Cancellation removes the waiter as well.
            notification.as_mut().enable();
            {
                let mut s = self.state.lock().unwrap();
                let free = self.total - s.used.iter().sum::<usize>();
                let protected: usize = s
                    .used
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != core && s.waiting[*i] > 0)
                    .map(|(_, used)| self.reserve.saturating_sub(*used))
                    .sum();
                if free > protected {
                    s.used[core] += 1;
                    s.waiting[core] -= 1;
                    waiting.active = false;
                    return Credit {
                        pool: self.clone(),
                        core,
                    };
                }
            }
            notification.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;

    #[tokio::test]
    async fn idle_reservation_is_borrowed_and_waiting_core_recovers_first_credit() {
        let pool = Arc::new(CreditPool::new(16, 2));
        let mut held = Vec::new();
        for _ in 0..16 {
            held.push(pool.acquire(0).await);
        }
        let waiting_small = pool.acquire(1);
        tokio::pin!(waiting_small);
        assert!(waiting_small.as_mut().now_or_never().is_none());
        drop(held.pop());
        let waiting_large = pool.acquire(0);
        tokio::pin!(waiting_large);
        assert!(waiting_large.as_mut().now_or_never().is_none());
        let small = waiting_small.await;
        assert_eq!(pool.state.lock().unwrap().used, vec![15, 1]);
        drop(small);
        let large = waiting_large.await;
        drop(large);
        drop(held);
        assert_eq!(pool.state.lock().unwrap().used, vec![0, 0]);
    }

    #[tokio::test]
    async fn cancelled_waiter_does_not_reserve_forever() {
        let pool = Arc::new(CreditPool::new(8, 2));
        let held: Vec<_> = futures::future::join_all((0..8).map(|_| pool.acquire(0))).await;
        {
            let mut waiting = Box::pin(pool.acquire(1));
            assert!(waiting.as_mut().now_or_never().is_none());
        }
        assert_eq!(pool.state.lock().unwrap().waiting, vec![0, 0]);
        drop(held);
        drop(pool.acquire(0).await);
    }
}
