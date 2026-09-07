//! One finite FIFO line cache per core. Port service is one cycle per lookup
//! and insertion. A loader credit retains returned data until its tile copy.
use runtime::{Duration, Executor};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::Semaphore;

pub const ENTRY_BYTES: usize = 80;

#[derive(Default)]
struct State {
    lines: BTreeMap<u64, [u8; 64]>,
    fifo: VecDeque<u64>,
}

pub struct ReadCache {
    entries: usize,
    clock: u64,
    port: Semaphore,
    state: Mutex<State>,
    pub requests: AtomicU64,
    pub hits: AtomicU64,
    pub busy: AtomicU64,
    pub peak: AtomicUsize,
}

impl ReadCache {
    pub fn new(bytes: usize, clock: u64) -> Self {
        Self {
            entries: bytes / ENTRY_BYTES,
            clock,
            port: Semaphore::new(1),
            state: Mutex::new(State::default()),
            requests: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            busy: AtomicU64::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    pub async fn lookup(&self, address: u64) -> Option<[u8; 64]> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        if self.entries == 0 {
            return None;
        }
        let _port = self.port.acquire().await.unwrap();
        self.busy.fetch_add(self.clock, Ordering::SeqCst);
        Executor::current()
            .resolve_at(Duration::from_picos(self.clock))
            .await;
        let value = self.state.lock().unwrap().lines.get(&address).copied();
        if value.is_some() {
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
        value
    }

    pub async fn insert(&self, address: u64, bytes: [u8; 64]) {
        if self.entries == 0 {
            return;
        }
        let _port = self.port.acquire().await.unwrap();
        self.busy.fetch_add(self.clock, Ordering::SeqCst);
        Executor::current()
            .resolve_at(Duration::from_picos(self.clock))
            .await;
        let mut state = self.state.lock().unwrap();
        if !state.lines.contains_key(&address) {
            if state.lines.len() == self.entries {
                let oldest = state.fifo.pop_front().unwrap();
                state.lines.remove(&oldest);
            }
            state.fifo.push_back(address);
        }
        state.lines.insert(address, bytes);
        self.peak
            .fetch_max(state.lines.len() * ENTRY_BYTES, Ordering::SeqCst);
    }
}
