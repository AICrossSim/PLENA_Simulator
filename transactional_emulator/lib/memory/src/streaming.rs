use std::sync::Mutex;

use runtime::Duration;
use tokio::sync::Semaphore;

use crate::MemoryModel;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightRegion {
    pub layer_id: u32,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightManifest {
    pub activation_ceiling: u64,
    pub regions: Vec<WeightRegion>,
}

impl WeightManifest {
    pub fn is_weight_addr(&self, addr: u64) -> bool {
        if addr < self.activation_ceiling {
            return false;
        }
        self.regions
            .iter()
            .any(|r| addr >= r.offset && addr < r.offset + r.size)
    }

    pub fn layer_for_addr(&self, addr: u64) -> Option<u32> {
        self.regions
            .iter()
            .find(|r| addr >= r.offset && addr < r.offset + r.size)
            .map(|r| r.layer_id)
    }
}

#[derive(Debug, Default)]
pub struct StreamingStats {
    pub weight_reads: u64,
    pub weight_bytes: u64,
    pub activation_reads: u64,
    pub activation_bytes: u64,
}

/// Direct host-to-SRAM streaming memory model.
///
/// Weight reads bypass DDR3 and model a host transfer at a fixed bandwidth.
/// Activation reads go through the inner memory model (DDR3 timing).
/// Data is always served from the backing store — only timing differs.
pub struct HostStream<T> {
    inner: T,
    manifest: WeightManifest,
    host_transfer_time_per_chunk: Duration,
    host_link: Semaphore,
    stats: Mutex<StreamingStats>,
}

impl<T> HostStream<T> {
    pub fn new(inner: T, manifest: WeightManifest, host_bandwidth_bytes_per_sec: u64) -> Self {
        let nanos_per_chunk = 64 * 1_000_000_000 / host_bandwidth_bytes_per_sec;
        Self {
            inner,
            manifest,
            host_transfer_time_per_chunk: Duration::from_nanos(nanos_per_chunk),
            host_link: Semaphore::new(1),
            stats: Mutex::new(StreamingStats::default()),
        }
    }

    pub fn statistics(&self) -> StreamingStats {
        let guard = self.stats.lock().unwrap();
        StreamingStats {
            weight_reads: guard.weight_reads,
            weight_bytes: guard.weight_bytes,
            activation_reads: guard.activation_reads,
            activation_bytes: guard.activation_bytes,
        }
    }
}

impl<T: MemoryModel> MemoryModel for HostStream<T> {
    async fn read(&self, addr: u64) -> [u8; 64] {
        if self.manifest.is_weight_addr(addr) {
            let _permit = self.host_link.acquire().await.unwrap();
            let rt = runtime::Executor::current();
            rt.resolve_at(self.host_transfer_time_per_chunk).await;
            {
                let mut stats = self.stats.lock().unwrap();
                stats.weight_reads += 1;
                stats.weight_bytes += 64;
            }
            self.inner.read(addr).await
        } else {
            {
                let mut stats = self.stats.lock().unwrap();
                stats.activation_reads += 1;
                stats.activation_bytes += 64;
            }
            self.inner.read(addr).await
        }
    }

    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        self.inner.write(addr, bytes).await
    }

    fn statistics_summary(&self, _elapsed_secs: f64) -> Option<String> {
        let s = self.statistics();
        Some(format!(
            "Host-stream memory - weight reads: {} ({} bytes) | activation reads: {} ({} bytes)",
            s.weight_reads, s.weight_bytes, s.activation_reads, s.activation_bytes
        ))
    }
}

#[derive(Debug, Default)]
pub struct LayerSwapStats {
    pub total_swaps: u64,
    pub total_swap_bytes: u64,
    pub total_swap_nanos: u64,
}

/// Layer-swapping DDR3 memory model.
///
/// Simulates a capacity-limited DDR3 where only a few layers' weights fit
/// at a time. When a weight read targets a non-resident layer, the model
/// charges a swap penalty (evict + load) based on host bandwidth.
/// Activation reads always go through DDR3 timing.
pub struct LayerSwapping<T> {
    inner: T,
    manifest: WeightManifest,
    capacity: u64,
    swap_bandwidth_bytes_per_sec: u64,
    state: Mutex<LayerSwapState>,
    stats: Mutex<LayerSwapStats>,
}

#[derive(Debug)]
struct LayerSwapState {
    resident_layers: Vec<u32>,
    used_bytes: u64,
}

impl<T> LayerSwapping<T> {
    pub fn new(
        inner: T,
        manifest: WeightManifest,
        capacity: u64,
        swap_bandwidth_bytes_per_sec: u64,
    ) -> Self {
        Self {
            inner,
            manifest,
            capacity,
            swap_bandwidth_bytes_per_sec,
            state: Mutex::new(LayerSwapState {
                resident_layers: Vec::new(),
                used_bytes: 0,
            }),
            stats: Mutex::new(LayerSwapStats::default()),
        }
    }

    pub fn statistics(&self) -> LayerSwapStats {
        let guard = self.stats.lock().unwrap();
        LayerSwapStats {
            total_swaps: guard.total_swaps,
            total_swap_bytes: guard.total_swap_bytes,
            total_swap_nanos: guard.total_swap_nanos,
        }
    }

    fn layer_size(&self, layer_id: u32) -> u64 {
        self.manifest
            .regions
            .iter()
            .filter(|r| r.layer_id == layer_id)
            .map(|r| r.size)
            .sum()
    }

    fn ensure_resident(&self, layer_id: u32) -> Option<Duration> {
        let mut state = self.state.lock().unwrap();
        if state.resident_layers.contains(&layer_id) {
            return None;
        }

        let needed = self.layer_size(layer_id);

        while state.used_bytes + needed > self.capacity && !state.resident_layers.is_empty() {
            let evicted = state.resident_layers.remove(0);
            let freed = self.layer_size(evicted);
            state.used_bytes = state.used_bytes.saturating_sub(freed);
        }

        let load_nanos = needed * 1_000_000_000 / self.swap_bandwidth_bytes_per_sec;
        state.resident_layers.push(layer_id);
        state.used_bytes += needed;

        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_swaps += 1;
            stats.total_swap_bytes += needed;
            stats.total_swap_nanos += load_nanos;
        }

        Some(Duration::from_nanos(load_nanos))
    }
}

impl<T: MemoryModel> MemoryModel for LayerSwapping<T> {
    async fn read(&self, addr: u64) -> [u8; 64] {
        if let Some(layer_id) = self.manifest.layer_for_addr(addr) {
            if let Some(swap_duration) = self.ensure_resident(layer_id) {
                let rt = runtime::Executor::current();
                rt.resolve_at(swap_duration).await;
            }
        }
        self.inner.read(addr).await
    }

    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        self.inner.write(addr, bytes).await
    }

    fn statistics_summary(&self, _elapsed_secs: f64) -> Option<String> {
        let s = self.statistics();
        Some(format!(
            "Layer-swap memory - swaps: {} | swapped bytes: {} | swap time: {} ns",
            s.total_swaps, s.total_swap_bytes, s.total_swap_nanos
        ))
    }
}
