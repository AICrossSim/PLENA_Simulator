//! Shared timing primitive for charged N3 and event-controller descriptors.
//! The optional second port has the same service width. Same-band RMWs still
//! serialize, conservatively covering context/band and bitmap hazards.
use super::*;
use tokio::sync::SemaphorePermit;

pub(super) struct ControlPort {
    ports: Semaphore,
    bands: Vec<Semaphore>,
    count: usize,
    intervals: Mutex<Vec<(u64, u64)>>, // observer only
}

pub(super) struct ControlLease<'a> {
    _port: SemaphorePermit<'a>,
    _band: Option<SemaphorePermit<'a>>,
    pub start: u64,
    pub duration: u64,
    pub wait: u64,
    pub conflict_wait: u64,
}

impl ControlPort {
    pub fn new(count: usize, bands: usize) -> Self {
        Self {
            ports: Semaphore::new(count),
            bands: (0..bands).map(|_| Semaphore::new(1)).collect(),
            count,
            intervals: Mutex::new(Vec::new()),
        }
    }

    pub async fn access(&self, duration: u64, band: usize) -> ControlLease<'_> {
        let ex = Executor::current();
        let begin = ex.now().as_picos();
        // Keep the frozen one-port queue exactly as it was. For two ports,
        // reserve the addressed band before occupying a service lane.
        let band_guard = if self.count > 1 {
            Some(self.bands[band].acquire().await.unwrap())
        } else {
            None
        };
        let conflict_wait = ex.now().as_picos() - begin;
        let port = self.ports.acquire().await.unwrap();
        let start = ex.now().as_picos();
        if duration > 0 {
            ex.resolve_at(Duration::from_picos(duration)).await;
            self.intervals
                .lock()
                .unwrap()
                .push((start, start + duration));
        }
        ControlLease {
            _port: port,
            _band: band_guard,
            start,
            duration,
            wait: start - begin,
            conflict_wait,
        }
    }

    /// Aggregate service counts both ports; occupancy is their interval union.
    pub fn occupancy(&self) -> (u64, usize) {
        let intervals = self.intervals.lock().unwrap();
        let mut edges: Vec<_> = intervals
            .iter()
            .flat_map(|&(s, e)| [(s, 1i32), (e, -1)])
            .collect();
        edges.sort_unstable(); // End before start at identical timestamps.
        let (mut last, mut active, mut peak, mut busy) = (0, 0i32, 0, 0);
        for (at, change) in edges {
            if active > 0 {
                busy += at - last;
            }
            active += change;
            assert!(active >= 0 && active <= self.count as i32);
            peak = peak.max(active as usize);
            last = at;
        }
        assert_eq!(active, 0);
        (busy, peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn two_lanes_overlap_distinct_bands_but_serialize_same_band_rmw() {
        for (ports, bands, expected_union, expected_peak) in [
            (1, [0, 1], 4000, 1),
            (2, [0, 1], 2000, 2),
            (2, [0, 0], 4000, 1),
        ] {
            let port = Arc::new(ControlPort::new(ports, 2));
            let ex = Executor::new();
            let finished = Arc::new(AtomicUsize::new(0));
            for band in bands {
                let (port, finished) = (port.clone(), finished.clone());
                ex.spawn(async move {
                    let _lease = port.access(2000, band).await;
                    finished.fetch_add(1, Ordering::SeqCst);
                });
            }
            ex.enter(Instant::ETERNITY).await;
            assert_eq!(finished.load(Ordering::SeqCst), 2);
            assert_eq!(port.occupancy(), (expected_union, expected_peak));
            assert_eq!(ex.now().as_picos(), expected_union);
        }
    }
}
