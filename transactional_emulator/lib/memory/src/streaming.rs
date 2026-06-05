use std::sync::Mutex;

use runtime::Duration;
use tokio::sync::Semaphore;

use crate::MemoryModel;

/// Convert a byte count into a transfer time in nanoseconds at a fixed
/// bandwidth. The multiplication is done in `u128` so large layers (where
/// `bytes * 1_000_000_000` would overflow `u64`) compute correctly.
fn transfer_nanos(bytes: u64, bandwidth_bytes_per_sec: u64) -> u64 {
    (bytes as u128 * 1_000_000_000u128 / bandwidth_bytes_per_sec as u128) as u64
}

/// Classification of a manifest region by the kind of tensor data it holds.
///
/// Used by the capacity-aware [`CapacityModel`] to decide which regions are
/// pinned in DDR, swapped under a capacity bound, or streamed from the host.
/// Defaults to `Weight` so legacy weight-only manifests (which omit the field)
/// still deserialize with every region treated as a weight region.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegionKind {
    #[default]
    Weight,
    Kv,
    Activation,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightRegion {
    pub layer_id: u32,
    pub offset: u64,
    pub size: u64,
    #[serde(default)]
    pub kind: RegionKind,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightManifest {
    pub activation_ceiling: u64,
    pub regions: Vec<WeightRegion>,
    /// Optional DDR capacity (bytes) the board exposes. When present it can be
    /// used as the capacity for regime auto-selection in place of a CLI flag.
    #[serde(default)]
    pub ddr_capacity_bytes: Option<u64>,
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

    /// Classify an address by the kind of the region containing it, or `None`
    /// if no region covers it.
    pub fn region_kind(&self, addr: u64) -> Option<RegionKind> {
        self.regions
            .iter()
            .find(|r| addr >= r.offset && addr < r.offset + r.size)
            .map(|r| r.kind)
    }

    /// Sum region sizes per kind, returning `(weight, kv, activation)` bytes.
    pub fn footprint_by_kind(&self) -> (u64, u64, u64) {
        let mut weight = 0u64;
        let mut kv = 0u64;
        let mut activation = 0u64;
        for r in &self.regions {
            match r.kind {
                RegionKind::Weight => weight += r.size,
                RegionKind::Kv => kv += r.size,
                RegionKind::Activation => activation += r.size,
            }
        }
        (weight, kv, activation)
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
        let nanos_per_chunk = transfer_nanos(64, host_bandwidth_bytes_per_sec);
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

        // A single layer that exceeds total capacity can never stay resident:
        // every access reloads it. We still admit/charge it (preserving the
        // timing semantics) but surface the condition so it isn't silent.
        if needed > self.capacity {
            tracing::warn!(
                "layer {layer_id} weights ({needed} bytes) exceed DDR3 capacity ({} bytes); modelling as reload on every access",
                self.capacity
            );
        }

        let load_nanos = transfer_nanos(needed, self.swap_bandwidth_bytes_per_sec);
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

/// DDR-capacity regime for a fixed-size board.
///
/// Selected explicitly on the CLI, or automatically by [`choose_regime`] from
/// the model's footprint relative to the board's DDR capacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regime {
    /// Everything fits in DDR: all kinds pinned resident, pure DDR timing.
    Resident,
    /// Weights fit in DDR (pinned); KV + activations are swapped under the
    /// remaining capacity, charging a host transfer on each miss.
    KvSwap,
    /// Weights do not fit in DDR: weights stream from the host link per access;
    /// KV + activations stay resident in DDR.
    WeightStream,
}

impl Regime {
    fn label(self) -> &'static str {
        match self {
            Regime::Resident => "resident",
            Regime::KvSwap => "kv-swap",
            Regime::WeightStream => "weight-stream",
        }
    }
}

/// Choose a DDR-capacity regime from a model's footprint.
///
/// * `total <= capacity`        → [`Regime::Resident`] (everything is pinned)
/// * else `weight <= capacity`  → [`Regime::KvSwap`] (weights pinned, KV swaps)
/// * else                       → [`Regime::WeightStream`] (weights stream)
pub fn choose_regime(weight_bytes: u64, total_bytes: u64, capacity: u64) -> Regime {
    if total_bytes <= capacity {
        Regime::Resident
    } else if weight_bytes <= capacity {
        Regime::KvSwap
    } else {
        Regime::WeightStream
    }
}

#[derive(Debug, Default)]
pub struct CapacityStats {
    /// Bytes served straight from DDR (pinned/resident) without a host transfer.
    pub resident_bytes: u64,
    /// Bytes charged as KV/activation swap-ins under the capacity bound.
    pub kv_swapped_bytes: u64,
    /// Number of KV/activation swap-in events (misses).
    pub swap_count: u64,
    /// Bytes charged as host-streamed weight chunks.
    pub weight_streamed_bytes: u64,
}

/// Swappable-region LRU residency for the KV-swap regime.
///
/// Keyed on `(offset, size)` regions (not layer ids) so KV and activation
/// regions are admitted/evicted independently of weights. Mirrors
/// [`LayerSwapping`]'s mechanics: a miss charges `transfer_nanos(size, host_bw)`
/// and the least-recently-admitted region is evicted when over capacity.
#[derive(Debug)]
struct SwapState {
    /// Resident regions as `(offset, size)`, oldest-first (LRU at index 0).
    resident: Vec<(u64, u64)>,
    used_bytes: u64,
}

/// Capacity-aware memory model for a fixed-size DDR board.
///
/// Wraps a DDR-timed inner model and applies a per-kind pinning policy chosen
/// by [`Regime`]. Data is always served from `inner`; only the extra host
/// transfer timing differs between regimes.
pub struct CapacityModel<T> {
    inner: T,
    manifest: WeightManifest,
    regime: Regime,
    /// DDR capacity available to swappable kinds (KV/activation) in `KvSwap`:
    /// `ddr_capacity - weight_footprint`, saturating at 0.
    swap_capacity: u64,
    host_bandwidth_bytes_per_sec: u64,
    /// Per-64B host-stream chunk time for the `WeightStream` regime.
    host_transfer_time_per_chunk: Duration,
    /// 1-permit link serialising host streaming (mirrors `HostStream`).
    host_link: Semaphore,
    swap_state: Mutex<SwapState>,
    stats: Mutex<CapacityStats>,
}

impl<T> CapacityModel<T> {
    pub fn new(
        inner: T,
        manifest: WeightManifest,
        ddr_capacity_bytes: u64,
        host_bandwidth_bytes_per_sec: u64,
        regime: Regime,
    ) -> Self {
        let (weight_footprint, _kv, _act) = manifest.footprint_by_kind();
        let swap_capacity = ddr_capacity_bytes.saturating_sub(weight_footprint);
        let nanos_per_chunk = transfer_nanos(64, host_bandwidth_bytes_per_sec);
        Self {
            inner,
            manifest,
            regime,
            swap_capacity,
            host_bandwidth_bytes_per_sec,
            host_transfer_time_per_chunk: Duration::from_nanos(nanos_per_chunk),
            host_link: Semaphore::new(1),
            swap_state: Mutex::new(SwapState {
                resident: Vec::new(),
                used_bytes: 0,
            }),
            stats: Mutex::new(CapacityStats::default()),
        }
    }

    pub fn statistics(&self) -> CapacityStats {
        let g = self.stats.lock().unwrap();
        CapacityStats {
            resident_bytes: g.resident_bytes,
            kv_swapped_bytes: g.kv_swapped_bytes,
            swap_count: g.swap_count,
            weight_streamed_bytes: g.weight_streamed_bytes,
        }
    }

    /// The `(offset, size)` of the region containing `addr`, if any.
    fn region_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        self.manifest
            .regions
            .iter()
            .find(|r| addr >= r.offset && addr < r.offset + r.size)
            .map(|r| (r.offset, r.size))
    }

    /// Ensure the swappable region containing `addr` is resident under the
    /// capacity bound. Returns the swap-in delay to charge on a miss, or `None`
    /// if already resident.
    fn ensure_swappable_resident(&self, addr: u64) -> Option<Duration> {
        let (offset, size) = self.region_bounds(addr)?;
        let mut state = self.swap_state.lock().unwrap();
        if state.resident.iter().any(|&(o, _)| o == offset) {
            return None;
        }

        while state.used_bytes + size > self.swap_capacity && !state.resident.is_empty() {
            let (_, freed) = state.resident.remove(0);
            state.used_bytes = state.used_bytes.saturating_sub(freed);
        }

        // A region larger than the whole swap capacity can never stay resident:
        // every access reloads it. Admit/charge it anyway (preserving timing).
        if size > self.swap_capacity {
            tracing::warn!(
                "KV/activation region at offset {offset} ({size} bytes) exceeds swap capacity ({} bytes); modelling as reload on every access",
                self.swap_capacity
            );
        }

        let load_nanos = transfer_nanos(size, self.host_bandwidth_bytes_per_sec);
        state.resident.push((offset, size));
        state.used_bytes += size;

        {
            let mut stats = self.stats.lock().unwrap();
            stats.swap_count += 1;
            stats.kv_swapped_bytes += size;
        }

        Some(Duration::from_nanos(load_nanos))
    }
}

impl<T: MemoryModel> MemoryModel for CapacityModel<T> {
    async fn read(&self, addr: u64) -> [u8; 64] {
        let kind = self.manifest.region_kind(addr);
        match self.regime {
            // Everything pinned in DDR: pure inner timing.
            Regime::Resident => {
                self.stats.lock().unwrap().resident_bytes += 64;
                self.inner.read(addr).await
            }
            // Weights pinned; KV + activations swap under the capacity bound.
            Regime::KvSwap => match kind {
                Some(RegionKind::Kv) | Some(RegionKind::Activation) => {
                    if let Some(swap_duration) = self.ensure_swappable_resident(addr) {
                        let rt = runtime::Executor::current();
                        rt.resolve_at(swap_duration).await;
                    }
                    self.inner.read(addr).await
                }
                // Weight regions / non-manifest / below ceiling → resident DDR.
                _ => {
                    self.stats.lock().unwrap().resident_bytes += 64;
                    self.inner.read(addr).await
                }
            },
            // Weights stream from the host link; KV + activations resident.
            Regime::WeightStream => match kind {
                Some(RegionKind::Weight) => {
                    let _permit = self.host_link.acquire().await.unwrap();
                    let rt = runtime::Executor::current();
                    rt.resolve_at(self.host_transfer_time_per_chunk).await;
                    self.stats.lock().unwrap().weight_streamed_bytes += 64;
                    self.inner.read(addr).await
                }
                _ => {
                    self.stats.lock().unwrap().resident_bytes += 64;
                    self.inner.read(addr).await
                }
            },
        }
    }

    async fn write(&self, addr: u64, bytes: [u8; 64]) {
        self.inner.write(addr, bytes).await
    }

    fn statistics_summary(&self, _elapsed_secs: f64) -> Option<String> {
        let s = self.statistics();
        Some(format!(
            "Capacity model [{}] - resident bytes: {} | kv swapped bytes: {} (swaps: {}) | weight streamed bytes: {}",
            self.regime.label(),
            s.resident_bytes,
            s.kv_swapped_bytes,
            s.swap_count,
            s.weight_streamed_bytes
        ))
    }
}
