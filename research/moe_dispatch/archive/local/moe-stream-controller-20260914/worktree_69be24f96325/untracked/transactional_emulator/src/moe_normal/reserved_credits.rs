//! Bounded proportional grants for the existing DMA lookup/response frontend.
//! The grant comparator is part of the existing lookup service, not a new
//! issue port. 32 B/core + 64 B shared state is charged in frontend_bytes.
use super::*;

struct State {
    used: Vec<usize>,
    waiting: Vec<usize>,
    reserved: Vec<u64>,
    cursor: usize,
}
pub(super) struct ReservedCreditPool {
    total: usize,
    state: Mutex<State>,
    changed: Notify,
    // FIFO heads use links in already paid fragment descriptors. Only the
    // head of each core competes; notification order is not arbitration.
    heads: Vec<Semaphore>,
}
pub(super) struct ReservedBytes {
    pool: Arc<ReservedCreditPool>,
    core: usize,
    bytes: u64,
}
pub(super) struct Credit {
    pool: Arc<ReservedCreditPool>,
    core: usize,
}
struct Waiting {
    pool: Arc<ReservedCreditPool>,
    core: usize,
    active: bool,
}
impl Drop for ReservedBytes {
    fn drop(&mut self) {
        self.pool.state.lock().unwrap().reserved[self.core] -= self.bytes;
        self.pool.changed.notify_waiters();
    }
}
impl Drop for Credit {
    fn drop(&mut self) {
        self.pool.state.lock().unwrap().used[self.core] -= 1;
        self.pool.changed.notify_waiters();
    }
}
impl Drop for Waiting {
    fn drop(&mut self) {
        if self.active {
            self.pool.state.lock().unwrap().waiting[self.core] -= 1;
            self.pool.changed.notify_waiters();
        }
    }
}
impl ReservedCreditPool {
    pub fn new(total: usize, cores: usize) -> Self {
        assert!(total > 0 && (1..=2).contains(&cores));
        Self {
            total,
            state: Mutex::new(State {
                used: vec![0; cores],
                waiting: vec![0; cores],
                reserved: vec![0; cores],
                cursor: 0,
            }),
            changed: Notify::new(),
            heads: (0..cores).map(|_| Semaphore::new(1)).collect(),
        }
    }
    pub fn reserve(self: &Arc<Self>, core: usize, bytes: u64) -> ReservedBytes {
        let mut s = self.state.lock().unwrap();
        s.reserved[core] = s.reserved[core]
            .checked_add(bytes)
            .expect("reserved weight bytes overflow");
        drop(s);
        self.changed.notify_waiters();
        ReservedBytes {
            pool: self.clone(),
            core,
            bytes,
        }
    }
    fn winner(s: &State) -> Option<usize> {
        (0..s.used.len())
            .map(|offset| (s.cursor + offset) % s.used.len())
            .filter(|&c| s.waiting[c] > 0 && s.reserved[c] > 0)
            .min_by(|&a, &b| {
                (s.used[a] as u128 * s.reserved[b] as u128)
                    .cmp(&(s.used[b] as u128 * s.reserved[a] as u128))
            })
    }
    pub async fn acquire(self: &Arc<Self>, core: usize) -> Credit {
        let _head = self.heads[core].acquire().await.unwrap();
        self.state.lock().unwrap().waiting[core] += 1;
        self.changed.notify_waiters();
        let mut waiting = Waiting {
            pool: self.clone(),
            core,
            active: true,
        };
        loop {
            let notification = self.changed.notified();
            tokio::pin!(notification);
            notification.as_mut().enable();
            {
                let mut s = self.state.lock().unwrap();
                if s.used.iter().sum::<usize>() < self.total && Self::winner(&s) == Some(core) {
                    s.used[core] += 1;
                    s.waiting[core] -= 1;
                    s.cursor = (core + 1) % s.used.len();
                    waiting.active = false;
                    drop(s);
                    self.changed.notify_waiters();
                    return Credit {
                        pool: self.clone(),
                        core,
                    };
                }
            }
            notification.await;
        }
    }
    pub fn assert_drained(&self) {
        assert!(self.heads.iter().all(|s| s.available_permits() == 1));
        let s = self.state.lock().unwrap();
        assert!(
            s.used.iter().all(|&v| v == 0)
                && s.waiting.iter().all(|&v| v == 0)
                && s.reserved.iter().all(|&v| v == 0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grants_follow_reserved_byte_ratio_and_rotate_exact_ties() {
        let mut s = State {
            used: vec![0, 0],
            waiting: vec![100, 100],
            reserved: vec![3000, 1000],
            cursor: 0,
        };
        for _ in 0..16 {
            let c = ReservedCreditPool::winner(&s).unwrap();
            s.used[c] += 1;
            s.cursor = (c + 1) % 2;
        }
        assert_eq!(s.used, vec![12, 4]);
        s.waiting[1] = 0;
        assert_eq!(ReservedCreditPool::winner(&s), Some(0));
        s.waiting[1] = 1;
        s.reserved[0] = 0;
        assert_eq!(ReservedCreditPool::winner(&s), Some(1));
    }
    #[tokio::test]
    async fn cancellation_and_idle_core_cannot_strand_credits_or_reserved_bytes() {
        let pool = Arc::new(ReservedCreditPool::new(4, 2));
        let reservation = pool.reserve(0, 128);
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(pool.acquire(0).await);
        }
        {
            let other = pool.reserve(1, 32);
            let mut waiter = Box::pin(pool.acquire(1));
            assert!(waiter.as_mut().now_or_never().is_none());
            drop(waiter);
            drop(other);
        }
        drop(held);
        drop(reservation);
        pool.assert_drained();
    }
}
