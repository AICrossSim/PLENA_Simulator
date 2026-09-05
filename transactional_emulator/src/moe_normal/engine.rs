use futures::{
    FutureExt,
    future::{BoxFuture, Shared as SharedFuture},
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use half::bf16;
use memory::ErasedMemoryModel;
use quantize::{DataType, FpType};
use runtime::{Duration, Executor, Instant};
use tokio::sync::{Semaphore, oneshot};

use super::dma_credits::CreditPool;
use super::read_cache::{ENTRY_BYTES, ReadCache};
use super::types::*;

const BLOCK: usize = 8;

fn checked(values: &[usize], what: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| format!("{what} overflows usize"))
    })
}

fn extra_cycles(a: &Architecture, c: &CoreConfig) -> u64 {
    if a.matrix_timing == MatrixTiming::LegacySerialized {
        return 0;
    }
    (usize::BITS - (c.mlen - 1).leading_zeros()) as u64 + a.mac_pipeline_cycles
}

fn service_cycles(a: &Architecture, c: &CoreConfig) -> u64 {
    match a.matrix_timing {
        MatrixTiming::Pipelined => {
            let supply = c.activation_elements_per_cycle.unwrap_or(c.mlen);
            (c.blen as u64).max((c.blen as u64 * c.mlen as u64).div_ceil(supply as u64))
        }
        MatrixTiming::LegacySerialized => c.mlen as u64 + a.mac_pipeline_cycles,
    }
}

fn pipeline_bytes(a: &Architecture, c: &CoreConfig) -> Result<usize, String> {
    checked(
        &[c.blen, extra_cycles(a, c) as usize, 4],
        "pipeline storage",
    )
}

fn weight_slot_bytes(c: &CoreConfig) -> Result<usize, String> {
    // BF16 decoded tile, packed E4M3 elements and packed E8M0 scales.
    checked(&[c.blen, c.mlen / BLOCK, 25], "weight slot")
}

fn vector_bytes(w: &Workload, rows: usize) -> Result<usize, String> {
    let per_row = checked(&[2, w.input_dim], "X and output")?
        .checked_add(checked(&[3, w.expert_hidden_dim], "gate/up/Z")?)
        .ok_or("vector storage overflow")?;
    checked(&[rows, per_row, 2], "BF16 vector storage")
}

fn accumulator_bytes(
    w: &Workload,
    rows: usize,
    a: &Architecture,
    c: &CoreConfig,
) -> Result<usize, String> {
    checked(
        &[rows, w.input_dim.max(w.expert_hidden_dim), 4],
        "accumulator",
    )?
    .checked_add(pipeline_bytes(a, c)?)
    .ok_or_else(|| "accumulator plus pipeline storage overflow".into())
}

fn region_valid(r: &MatrixRegion, n: usize, k: usize, len: u64) -> Result<(), String> {
    if r.rows != n || r.cols != k {
        return Err(format!(
            "matrix shape is [{},{}], expected [{n},{k}]",
            r.rows, r.cols
        ));
    }
    let blocks = k.div_ceil(BLOCK) as u64;
    let padded_k = blocks
        .checked_mul(BLOCK as u64)
        .ok_or("padded K overflow")?;
    if r.element_row_stride < padded_k || r.scale_row_stride < blocks {
        return Err("MX row strides must cover K padded to block 8 and all row scales".into());
    }
    for (name, base, stride, width) in [
        ("element", r.element_base, r.element_row_stride, padded_k),
        ("scale", r.scale_base, r.scale_row_stride, blocks),
    ] {
        let end = (n as u64 - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(base))
            .and_then(|v| v.checked_add(width))
            .ok_or_else(|| format!("{name} address overflow"))?;
        if end > len {
            return Err(format!(
                "{name} matrix ends at {end}, beyond HBM image length {len}"
            ));
        }
    }
    Ok(())
}

#[derive(Clone)]
struct Row {
    token: usize,
    output: usize,
}

#[derive(Clone)]
struct Job {
    id: usize,
    expert: usize,
    shared: bool,
    rows: Vec<Row>,
}

struct Plan {
    queues: Vec<Vec<Job>>,
    routes: Vec<Route>,
    output_rows: usize,
    shared_base: usize,
    global_storage: usize,
}

fn job_fits(w: &Workload, a: &Architecture, c: &CoreConfig, job: &Job) -> bool {
    vector_bytes(w, job.rows.len()).is_ok_and(|v| v <= c.vector_sram_bytes)
        && accumulator_bytes(w, job.rows.len(), a, c).is_ok_and(|v| v <= c.accumulator_bytes)
}

fn plan(w: &Workload, a: &Architecture) -> Result<Plan, String> {
    let mut routes = w.routes.clone();
    routes.sort_by_key(|r| (r.token, r.slot));
    let mut grouped: BTreeMap<usize, Vec<Row>> = BTreeMap::new();
    for (i, r) in routes.iter().enumerate() {
        grouped.entry(r.expert).or_default().push(Row {
            token: r.token,
            output: i,
        });
    }
    let mut jobs: Vec<Job> = grouped
        .into_iter()
        .enumerate()
        .map(|(id, (expert, rows))| Job {
            id,
            expert,
            shared: false,
            rows,
        })
        .collect();
    let shared_base = routes.len();
    let output_rows = routes
        .len()
        .checked_add(if w.shared_expert.is_some() {
            w.inputs_bf16.len()
        } else {
            0
        })
        .ok_or("route count overflow")?;
    if let Some(shared) = &w.shared_expert {
        jobs.push(Job {
            id: jobs.len(),
            expert: shared.expert,
            shared: true,
            rows: (0..w.inputs_bf16.len())
                .map(|t| Row {
                    token: t,
                    output: shared_base + t,
                })
                .collect(),
        });
    }
    // Inputs and final BF16 output each need 2 bytes, final FP32 sum needs 4.
    // Every route result is BF16, held in a finite reorder region until combine.
    let global_storage = checked(
        &[w.inputs_bf16.len(), w.input_dim, 8],
        "shared inputs/output/sum",
    )?
    .checked_add(checked(
        &[output_rows, w.input_dim, 2],
        "route reorder storage",
    )?)
    .ok_or("shared storage overflow")?;
    if global_storage > a.combine_sram_bytes {
        return Err(format!(
            "shared input/combine SRAM needs {global_storage} bytes, capacity {}",
            a.combine_sram_bytes
        ));
    }
    if checked(&[jobs.len(), 64], "ready-job descriptors")? > a.dispatch_queue_bytes {
        return Err("ready-job descriptors exceed dispatch_queue_bytes".into());
    }
    let mut queues = vec![Vec::new(); a.cores.len()];
    for job in jobs {
        let mut target = if job.rows.len() >= a.dispatch_threshold {
            a.large_core
        } else {
            a.small_core
        };
        if a.dispatch_policy == DispatchPolicy::WorkConserving
            && !job_fits(w, a, &a.cores[target], &job)
        {
            target = a
                .cores
                .iter()
                .position(|c| job_fits(w, a, c, &job))
                .ok_or("expert group does not fit any core; group spilling is not enabled")?;
        }
        let c = &a.cores[target];
        let v = vector_bytes(w, job.rows.len())?;
        let acc = accumulator_bytes(w, job.rows.len(), a, c)?;
        if v > c.vector_sram_bytes {
            return Err(format!(
                "core {} expert {} M={} requires {v} vector SRAM bytes, capacity {}; V0 does not spill or tile groups",
                c.id,
                job.expert,
                job.rows.len(),
                c.vector_sram_bytes
            ));
        }
        if acc > c.accumulator_bytes {
            return Err(format!(
                "core {} expert {} M={} requires {acc} accumulator/pipeline bytes, capacity {}",
                c.id,
                job.expert,
                job.rows.len(),
                c.accumulator_bytes
            ));
        }
        queues[target].push(job);
    }
    Ok(Plan {
        queues,
        routes,
        output_rows,
        shared_base,
        global_storage,
    })
}

fn validate_model_bounds(w: &Workload, a: &Architecture, p: &Plan) -> Result<(), String> {
    // Reject impossible counter/timer ranges before entering the executor.
    // This bounds our own computation and vector work; backend HBM timing is
    // still the backend's contract. u128 keeps the validation itself checked.
    let lanes = a.vector_elements_per_cycle;
    let mut cycles =
        checked(&[p.output_rows, w.input_dim], "combine elements")?.div_ceil(lanes) as u128;
    cycles +=
        checked(&[w.inputs_bf16.len(), w.input_dim], "output elements")?.div_ceil(lanes) as u128;
    let mut issued = 0u128;
    let mut multipliers = 0u128;
    for (c, queue) in a.cores.iter().zip(&p.queues) {
        multipliers += checked(&[c.blen, c.mlen], "multipliers")? as u128;
        let per_tile_macs = checked(&[c.blen, c.blen, c.mlen], "tile MACs")? as u128;
        let tile_latency = service_cycles(a, c) as u128 + extra_cycles(a, c) as u128;
        let decode_cycles = checked(&[c.blen, c.mlen], "decode elements")?.div_ceil(lanes) as u128;
        let candidates: Vec<&Job> = if a.dispatch_policy == DispatchPolicy::WorkConserving {
            p.queues
                .iter()
                .flatten()
                .filter(|j| job_fits(w, a, c, j))
                .collect()
        } else {
            queue.iter().collect()
        };
        for job in candidates {
            cycles = cycles
                .checked_add(a.dispatch_cycles as u128)
                .ok_or("dispatch timing overflow")?;
            let m = job.rows.len();
            cycles +=
                2 * checked(&[m, w.input_dim], "gather/copy elements")?.div_ceil(lanes) as u128;
            cycles +=
                checked(&[m, w.expert_hidden_dim], "SwiGLU elements")?.div_ceil(lanes) as u128;
            for (n, k) in [
                (w.expert_hidden_dim, w.input_dim),
                (w.expert_hidden_dim, w.input_dim),
                (w.input_dim, w.expert_hidden_dim),
            ] {
                let weight_tiles = checked(
                    &[n.div_ceil(c.blen), k.div_ceil(c.mlen)],
                    "weight tile count",
                )? as u128;
                let macro_tiles = weight_tiles
                    .checked_mul(m.div_ceil(c.blen) as u128)
                    .ok_or("macro tile count overflow")?;
                issued = issued
                    .checked_add(
                        macro_tiles
                            .checked_mul(per_tile_macs)
                            .ok_or("issued MAC overflow")?,
                    )
                    .ok_or("issued MAC overflow")?;
                let upper = macro_tiles
                    .checked_mul(tile_latency)
                    .and_then(|v| {
                        weight_tiles
                            .checked_mul(decode_cycles)
                            .and_then(|decode| v.checked_add(decode))
                    })
                    .ok_or("timing bound overflow")?;
                cycles = cycles.checked_add(upper).ok_or("timing bound overflow")?;
                // Conservative: at most one line request per payload byte,
                // one lookup and one fill cycle per request.
                if c.read_cache_bytes > 0 {
                    let cache_cycles = weight_tiles
                        .checked_mul(per_tile_macs)
                        .and_then(|v| v.checked_mul(4))
                        .ok_or("cache timing overflow")?;
                    cycles = cycles
                        .checked_add(cache_cycles)
                        .ok_or("cache timing overflow")?;
                }
            }
        }
    }
    if issued > u64::MAX as u128 || multipliers > u64::MAX as u128 {
        return Err("workload/architecture exceeds u64 MAC counter range".into());
    }
    if cycles
        .checked_mul(a.clock_period_ps as u128)
        .is_none_or(|t| t > u64::MAX as u128)
    {
        return Err(
            "workload/architecture computation or vector duration exceeds u64 picoseconds".into(),
        );
    }
    Ok(())
}

/// Validate all capacities, shape/route contracts and complete HBM ranges before
/// spawning work. Out-of-range MemoryBacked reads must never silently turn to 0.
pub fn validate(w: &Workload, a: &Architecture, hbm_len: u64) -> Result<(), String> {
    if w.schema_version != 1 || a.schema_version != 1 {
        return Err("only workload/architecture schema_version 1 is supported".into());
    }
    if w.input_dim == 0 || w.expert_hidden_dim == 0 || w.inputs_bf16.is_empty() {
        return Err("input_dim, expert_hidden_dim and token count must be positive".into());
    }
    if !hbm_len.is_multiple_of(64) || hbm_len == 0 {
        return Err("HBM image length must be a positive multiple of 64 bytes".into());
    }
    if !(1..=2).contains(&a.cores.len())
        || a.large_core >= a.cores.len()
        || a.small_core >= a.cores.len()
    {
        return Err("V0 supports one or two cores, with in-range dispatch core indices".into());
    }
    if a.cores.len() == 2 && a.large_core == a.small_core {
        return Err("a two-core architecture must select distinct large/small cores".into());
    }
    if a.clock_period_ps == 0
        || a.vector_elements_per_cycle == 0
        || a.dispatch_threshold == 0
        || a.global_dma_credits == 0
    {
        return Err("clock, vector throughput, threshold and DMA credits must be positive".into());
    }
    if a.global_dma_credits > Semaphore::MAX_PERMITS
        || checked(&[a.global_dma_credits, 64], "DMA staging")? > a.global_dma_staging_bytes
    {
        return Err("global DMA staging must provide 64 bytes per credit".into());
    }
    if let Some(dma) = &a.dma {
        if !(1..=2).contains(&dma.lookup_ii_cycles) {
            return Err("lookup_ii_cycles must be 1 or 2".into());
        }
        if dma.fair_credits && a.global_dma_credits < 8 {
            return Err("fair DMA credits require at least 8 credits".into());
        }
        if a.global_dma_credits > 128 {
            return Err("DMA V1 supports at most 128 line/waiter credits".into());
        }
        if a.cores.iter().any(|c| c.read_cache_bytes != 0) {
            return Err("DMA V1 uses element bypass; legacy read cache must be disabled".into());
        }
        if frontend_bytes(a)? > dma.frontend_sram_bytes {
            return Err("DMA frontend metadata/response storage exceeds capacity".into());
        }
    }
    let mut ids = BTreeSet::new();
    for c in &a.cores {
        if c.activation_elements_per_cycle == Some(0) {
            return Err("activation supply must be positive".into());
        }
        if c.read_cache_bytes > 0 && c.read_cache_bytes < ENTRY_BYTES {
            return Err("read cache needs at least one 80-byte data/tag entry".into());
        }
        if c.id.is_empty()
            || !ids.insert(c.id.clone())
            || c.blen == 0
            || c.mlen == 0
            || !c.mlen.is_multiple_of(BLOCK)
        {
            return Err(
                "core ids must be unique/nonempty; BLEN positive and MLEN a positive multiple of 8"
                    .into(),
            );
        }
        // Keep the legacy square MatrixMachine's complete BLEN column slices
        // inside an MLEN row. The local normal-buffer loop itself could handle
        // other ratios; those would be a separate mapping/ISA extension.
        if !c.mlen.is_multiple_of(c.blen) {
            return Err(format!(
                "core {} MLEN must be divisible by BLEN for the square mapping",
                c.id
            ));
        }
        checked(&[c.blen, c.mlen], "multiplier count")?;
        checked(&[c.blen, c.blen, c.mlen], "issued MACs per tile")?;
        if a.mac_pipeline_cycles > u32::MAX as u64 {
            return Err("mac_pipeline_cycles exceeds supported range".into());
        }
        if !(2..=4).contains(&c.weight_slots) {
            return Err("weight_slots must be 2, 3 or 4".into());
        }
        let need = weight_slot_bytes(c)?
            .checked_mul(c.weight_slots)
            .ok_or("weight SRAM overflow")?;
        if need > c.weight_sram_bytes {
            return Err(format!(
                "core {} needs {need} bytes for configured MX/BF16 weight slots, capacity {}",
                c.id, c.weight_sram_bytes
            ));
        }
        let cycles = service_cycles(a, c)
            .checked_add(extra_cycles(a, c))
            .ok_or("tile timing overflow")?;
        cycles
            .checked_mul(a.clock_period_ps)
            .ok_or("tile duration overflow")?;
        pipeline_bytes(a, c)?;
    }
    for row in &w.inputs_bf16 {
        if row.len() != w.input_dim
            || row
                .iter()
                .any(|v| !bf16::from_bits(*v).to_f32().is_finite())
        {
            return Err("inputs must contain input_dim finite BF16 values per token".into());
        }
    }
    let mut experts = BTreeSet::new();
    for e in &w.experts {
        if !experts.insert(e.id) {
            return Err(format!("duplicate expert id {}", e.id));
        }
        region_valid(&e.gate, w.expert_hidden_dim, w.input_dim, hbm_len)?;
        region_valid(&e.up, w.expert_hidden_dim, w.input_dim, hbm_len)?;
        region_valid(&e.down, w.input_dim, w.expert_hidden_dim, hbm_len)?;
    }
    let mut slots = BTreeSet::new();
    for r in &w.routes {
        if r.token >= w.inputs_bf16.len() || !experts.contains(&r.expert) || !r.weight.is_finite() {
            return Err("route references an invalid token/expert or non-finite weight".into());
        }
        if !slots.insert((r.token, r.slot)) {
            return Err(format!("duplicate route slot ({},{})", r.token, r.slot));
        }
    }
    if let Some(s) = &w.shared_expert
        && (!experts.contains(&s.expert) || !s.weight.is_finite())
    {
        return Err("shared expert reference or weight is invalid".into());
    }
    let p = plan(w, a)?;
    validate_model_bounds(w, a, &p)?;
    Ok(())
}

struct CoreState {
    cache: ReadCache,
    config: CoreConfig,
    report: Mutex<CoreReport>,
    slots: AtomicUsize,
    slots_peak: AtomicUsize,
    weight_bytes: AtomicUsize,
    weight_peak: AtomicUsize,
}

impl CoreState {
    fn new(c: &CoreConfig, a: &Architecture) -> Self {
        Self {
            cache: ReadCache::new(c.read_cache_bytes, a.clock_period_ps),
            config: c.clone(),
            report: Mutex::new(CoreReport {
                id: c.id.clone(),
                blen: c.blen,
                mlen: c.mlen,
                multipliers: (c.blen * c.mlen) as u64,
                jobs: 0,
                useful_macs: 0,
                issued_macs: 0,
                compute_busy_ps: 0,
                accumulator_dependency_stall_ps: 0,
                pipeline_drain_ps: 0,
                pipeline_register_bytes: pipeline_bytes(a, c).unwrap(),
                weight_ready_wait_ps: 0,
                vector_wait_ps: 0,
                vector_sram_peak_bytes: 0,
                accumulator_peak_bytes: 0,
                weight_sram_peak_bytes: 0,
                weight_slots_peak: 0,
                hbm_read_bytes: 0,
                compute_busy_fraction: 0.0,
                mac_utilization: 0.0,
                cache_requests: 0,
                cache_hits: 0,
                cache_port_busy_ps: 0,
                cache_peak_bytes: 0,
            }),
            slots: AtomicUsize::new(0),
            slots_peak: AtomicUsize::new(0),
            weight_bytes: AtomicUsize::new(0),
            weight_peak: AtomicUsize::new(0),
        }
    }
}

struct SlotReservation {
    core: Arc<CoreState>,
    bytes: usize,
}

impl SlotReservation {
    fn new(core: Arc<CoreState>) -> Self {
        let bytes = weight_slot_bytes(&core.config).unwrap();
        let slots = core.slots.fetch_add(1, Ordering::SeqCst) + 1;
        let occupied = core.weight_bytes.fetch_add(bytes, Ordering::SeqCst) + bytes;
        assert!(
            slots <= core.config.weight_slots && occupied <= core.config.weight_sram_bytes,
            "weight slot ownership violated"
        );
        core.slots_peak.fetch_max(slots, Ordering::SeqCst);
        core.weight_peak.fetch_max(occupied, Ordering::SeqCst);
        Self { core, bytes }
    }
}

impl Drop for SlotReservation {
    fn drop(&mut self) {
        self.core.slots.fetch_sub(1, Ordering::SeqCst);
        self.core
            .weight_bytes
            .fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

struct Dma {
    hbm: Arc<dyn ErasedMemoryModel>,
    credits: Semaphore,
    fair_pool: Option<Arc<CreditPool>>,
    core_indices: BTreeMap<String, usize>,
    inflight: AtomicUsize,
    peak: AtomicUsize,
    bytes: AtomicU64,
    config: Option<DmaConfig>,
    clock: u64,
    lookup: Vec<Semaphore>,
    copy: Vec<Semaphore>,
    lines: Mutex<BTreeMap<u64, PendingLine>>,
    report: Mutex<DmaReport>,
}

// Conservative physical reservation, independent of host future/Vec sizes.
// Per line: 64B response + 32B MSHR + 24B waiter. Native pool: 256 * 16B.
// Tile copy fragments cost 24B each; 128B state per resident tile. 4KiB
// covers queue heads, lookup/copy pipelines, and bounded arbitration state.
fn frontend_bytes(a: &Architecture) -> Result<usize, String> {
    let mut bytes = checked(&[a.global_dma_credits, 120], "DMA line metadata")?
        .checked_add(8192)
        .ok_or("DMA control storage overflow")?;
    for c in &a.cores {
        let fragments = c
            .mlen
            .div_ceil(64)
            .checked_add(c.mlen.div_ceil(BLOCK).div_ceil(64))
            .and_then(|v| v.checked_add(2))
            .ok_or("DMA fragment count overflow")?;
        let tile = checked(&[c.blen, fragments, 24], "DMA fragment descriptors")?
            .checked_add(128)
            .ok_or("DMA tile state overflow")?;
        bytes = bytes
            .checked_add(checked(
                &[c.weight_slots, tile],
                "DMA resident descriptors",
            )?)
            .ok_or("DMA frontend storage overflow")?;
    }
    Ok(bytes)
}

type SectorFuture = SharedFuture<BoxFuture<'static, [u8; 32]>>;
struct PendingLine {
    users: usize,
    sectors: [Option<SectorFuture>; 2],
}

impl Dma {
    fn sector(self: &Arc<Self>, address: u64, sector: usize, core: Arc<CoreState>) -> SectorFuture {
        let dma = self.clone();
        async move {
            let data = dma.hbm.box_read_mask(address, 1 << sector).await;
            dma.bytes.fetch_add(32, Ordering::SeqCst);
            core.report.lock().unwrap().hbm_read_bytes += 32;
            let mut bytes = [0; 32];
            bytes.copy_from_slice(&data[sector * 32..sector * 32 + 32]);
            bytes
        }
        .boxed()
        .shared()
    }

    async fn read(
        self: &Arc<Self>,
        address: u64,
        requested: u8,
        useful: usize,
        core: Arc<CoreState>,
    ) -> [u8; 64] {
        let config = self.config.as_ref().unwrap();
        let mask = if config.sector_reads { requested } else { 3 };
        let ex = Executor::current();
        {
            // Four banks, conservative 2-cycle occupied lookup, common to all
            // DMA V1 policies including global-FIFO and coalescing-off controls.
            let _port = self.lookup[(address as usize / 64) % 4]
                .acquire()
                .await
                .unwrap();
            // Two-cycle latency with an independently pipelined lookup port.
            ex.resolve_at(Duration::from_picos(
                config.lookup_ii_cycles as u64 * self.clock,
            ))
            .await;
            drop(_port);
            if config.lookup_ii_cycles == 1 {
                ex.resolve_at(Duration::from_picos(self.clock)).await;
            }
            let mut r = self.report.lock().unwrap();
            r.lookup_busy_ps += config.lookup_ii_cycles as u64 * self.clock;
            r.line_requests += 1;
            r.sector_requests += u64::from(mask.count_ones());
            r.useful_copy_bytes += useful as u64;
        }
        let sectors = if config.coalesce {
            let mut lines = self.lines.lock().unwrap();
            let entry = lines.entry(address).or_insert_with(|| PendingLine {
                users: 0,
                sectors: [None, None],
            });
            entry.users += 1;
            let mut needed = Vec::new();
            for sector in 0..2 {
                if mask & (1 << sector) == 0 {
                    continue;
                }
                if entry.sectors[sector].is_some() {
                    self.report.lock().unwrap().merged_sectors += 1;
                } else {
                    entry.sectors[sector] = Some(self.sector(address, sector, core.clone()));
                }
                needed.push((sector, entry.sectors[sector].as_ref().unwrap().clone()));
            }
            let mut report = self.report.lock().unwrap();
            report.mshr_peak = report.mshr_peak.max(lines.len());
            needed
        } else {
            (0..2)
                .filter(|s| mask & (1 << s) != 0)
                .map(|s| (s, self.sector(address, s, core.clone())))
                .collect()
        };
        let responses = futures::future::join_all(
            sectors
                .into_iter()
                .map(|(index, future)| async move { (index, future.await) }),
        )
        .await;
        let mut bytes = [0; 64];
        for (index, data) in responses {
            bytes[index * 32..index * 32 + 32].copy_from_slice(&data);
        }
        {
            // Eight return/copy banks, 32 B per cycle each. Duplicate consumers
            // each pay copy service; multicast and SRAM writes are not free.
            let _port = self.copy[(address as usize / 64) % 8]
                .acquire()
                .await
                .unwrap();
            let duration = useful.div_ceil(32) as u64 * self.clock;
            ex.resolve_at(Duration::from_picos(duration)).await;
            self.report.lock().unwrap().copy_busy_ps += duration;
        }
        if config.coalesce {
            let mut lines = self.lines.lock().unwrap();
            let entry = lines.get_mut(&address).unwrap();
            entry.users -= 1;
            if entry.users == 0 {
                lines.remove(&address);
            }
        }
        bytes
    }
}

struct VectorUnit {
    permit: Semaphore,
    lanes: usize,
    clock: u64,
    busy_ps: AtomicU64,
}

impl VectorUnit {
    async fn work(&self, elements: usize) -> u64 {
        if elements == 0 {
            return 0;
        }
        let ex = Executor::current();
        let begin = ex.now().as_picos();
        let _permit = self.permit.acquire().await.unwrap();
        let waiting = ex.now().as_picos() - begin;
        let duration = elements.div_ceil(self.lanes) as u64 * self.clock;
        self.busy_ps.fetch_add(duration, Ordering::SeqCst);
        ex.resolve_at(Duration::from_picos(duration)).await;
        waiting
    }
}

struct Shared {
    ready: Mutex<Vec<Job>>,
    dispatcher: Semaphore,
    dispatcher_busy: AtomicU64,
    dma: Arc<Dma>,
    vector: Arc<VectorUnit>,
    reorder: Mutex<Vec<bf16>>,
    completions: Mutex<Vec<JobCompletion>>,
}

#[derive(Clone, Copy)]
struct TileSpec {
    n: usize,
    k: usize,
}

struct WeightTile {
    values: Vec<bf16>,
    _reservation: SlotReservation,
}

#[derive(Clone, Copy)]
struct CopySpan {
    src: usize,
    dst: usize,
    len: usize,
}

fn add_reads(reads: &mut BTreeMap<u64, Vec<CopySpan>>, base: u64, len: usize, dst: usize) {
    let mut offset = 0;
    while offset < len {
        let addr = base + offset as u64;
        let aligned = addr / 64 * 64;
        let src = (addr % 64) as usize;
        let count = (64 - src).min(len - offset);
        reads.entry(aligned).or_default().push(CopySpan {
            src,
            dst: dst + offset,
            len: count,
        });
        offset += count;
    }
}

fn spawn_load(
    core: Arc<CoreState>,
    shared: Arc<Shared>,
    region: MatrixRegion,
    tile: TileSpec,
) -> oneshot::Receiver<Result<WeightTile, String>> {
    let reservation = SlotReservation::new(core.clone());
    let (tx, rx) = oneshot::channel();
    Executor::current().spawn(async move {
        let result = load_tile(core, shared, region, tile, reservation).await;
        let _ = tx.send(result);
    });
    rx
}

async fn load_tile(
    core: Arc<CoreState>,
    shared: Arc<Shared>,
    region: MatrixRegion,
    tile: TileSpec,
    reservation: SlotReservation,
) -> Result<WeightTile, String> {
    let c = &core.config;
    let nr = c.blen.min(region.rows - tile.n);
    let kr = c.mlen.min(region.cols - tile.k);
    let elements = c.blen * c.mlen;
    let scale_count = elements / BLOCK;
    let packed = Arc::new(Mutex::new(vec![0u8; elements + scale_count]));
    let mut reads = BTreeMap::new();
    for row in 0..nr {
        add_reads(
            &mut reads,
            region.element_base + (tile.n + row) as u64 * region.element_row_stride + tile.k as u64,
            kr,
            row * c.mlen,
        );
        add_reads(
            &mut reads,
            region.scale_base
                + (tile.n + row) as u64 * region.scale_row_stride
                + (tile.k / BLOCK) as u64,
            kr.div_ceil(BLOCK),
            elements + row * (c.mlen / BLOCK),
        );
    }
    let mut pending = Vec::with_capacity(reads.len());
    for (address, spans) in reads {
        let dma = shared.dma.clone();
        let packed = packed.clone();
        let core = core.clone();
        let (tx, rx) = oneshot::channel();
        pending.push(rx);
        Executor::current().spawn(async move {
            let (credit, fair_credit) = if let Some(pool) = &dma.fair_pool {
                let begin = Executor::current().now().as_picos();
                let credit = pool.acquire(dma.core_indices[&core.config.id]).await;
                dma.report.lock().unwrap().fair_credit_wait_ps +=
                    Executor::current().now().as_picos() - begin;
                (None, Some(credit))
            } else {
                (Some(dma.credits.acquire().await.unwrap()), None)
            };
            let current = dma.inflight.fetch_add(1, Ordering::SeqCst) + 1;
            dma.peak.fetch_max(current, Ordering::SeqCst);
            let bytes = if dma.config.is_some() {
                let mut mask = 0u8;
                for span in &spans {
                    mask |= 1 << (span.src / 32);
                    mask |= 1 << ((span.src + span.len - 1) / 32);
                }
                dma.read(
                    address,
                    mask,
                    spans.iter().map(|s| s.len).sum(),
                    core.clone(),
                )
                .await
            } else if let Some(bytes) = core.cache.lookup(address).await {
                bytes
            } else {
                let bytes = dma.hbm.box_read(address).await;
                dma.bytes.fetch_add(64, Ordering::SeqCst);
                core.report.lock().unwrap().hbm_read_bytes += 64;
                core.cache.insert(address, bytes).await;
                bytes
            };
            {
                let mut buffer = packed.lock().unwrap();
                for s in spans {
                    buffer[s.dst..s.dst + s.len].copy_from_slice(&bytes[s.src..s.src + s.len]);
                }
            }
            dma.inflight.fetch_sub(1, Ordering::SeqCst);
            // Credit covers both the HBM request and its 64-byte staging data.
            drop(credit);
            drop(fair_credit);
            let _ = tx.send(());
        });
    }
    for rx in pending {
        rx.await.map_err(|_| "DMA request task failed")?;
    }
    // MX decode and BF16 placement use the same finite shared vector actor;
    // decoded data cannot become available for free at the final HBM response.
    let wait = shared.vector.work(nr * kr).await;
    core.report.lock().unwrap().vector_wait_ps += wait;
    let packed = Arc::try_unwrap(packed)
        .map_err(|_| "DMA buffer retained after completion")?
        .into_inner()
        .unwrap();
    // Reuse the actual repository codec, including its local E4M3 subnormal rule.
    // Decode one scalar at a time, avoiding an unaccounted full FP32 tile.
    let element_type = DataType::Fp(FpType {
        sign: true,
        exponent: 4,
        mantissa: 3,
    });
    let scale_type = DataType::Fp(FpType::E8M0);
    let mut values = vec![bf16::ZERO; elements];
    for row in 0..nr {
        for k in 0..kr {
            let element = element_type.convert_bits_to_f32(packed[row * c.mlen + k] as u32);
            let scale = scale_type
                .convert_bits_to_f32(packed[elements + row * (c.mlen / BLOCK) + k / BLOCK] as u32);
            let v = element * scale;
            let value = bf16::from_f32(v);
            if !value.is_finite() {
                return Err("decoded weight is not finite BF16".into());
            }
            values[row * c.mlen + k] = value;
        }
    }
    Ok(WeightTile {
        values,
        _reservation: reservation,
    })
}

// Keep the three separately owned data buffers explicit at the execution boundary.
#[allow(clippy::too_many_arguments)]
async fn gemm(
    input: &[bf16],
    output: &mut [bf16],
    accumulator: &mut [f32],
    m: usize,
    region: &MatrixRegion,
    core: &Arc<CoreState>,
    a: &Architecture,
    shared: &Arc<Shared>,
) -> Result<(), String> {
    let n = region.rows;
    let k = region.cols;
    let c = &core.config;
    let ex = Executor::current();
    let acc = &mut accumulator[..m * n];
    acc.fill(0.0);
    let mb = m.div_ceil(c.blen);
    let nb = n.div_ceil(c.blen);
    // One readiness time per physical M,N output block. The complete output
    // accumulator and fixed in-flight result registers were already reserved.
    let mut ready = vec![ex.now(); mb * nb];
    let mut specs = Vec::new();
    for n0 in (0..n).step_by(c.blen) {
        for k0 in (0..k).step_by(c.mlen) {
            specs.push(TileSpec { n: n0, k: k0 });
        }
    }
    let mut pending = VecDeque::new();
    let mut issued = 0;
    // Initial window reserves every slot before DMA. At each retirement exactly
    // one slot becomes reusable; READY tiles keep their ownership meanwhile.
    for spec in specs.iter().take(c.weight_slots) {
        pending.push_back(spawn_load(
            core.clone(),
            shared.clone(),
            region.clone(),
            *spec,
        ));
        issued += 1;
    }
    for spec in specs.iter().copied() {
        let wait_start = ex.now().as_picos();
        let tile = pending
            .pop_front()
            .unwrap()
            .await
            .map_err(|_| "weight loader task failed")??;
        core.report.lock().unwrap().weight_ready_wait_ps += ex.now().as_picos() - wait_start;
        for m0 in (0..m).step_by(c.blen) {
            let block_index = (spec.n / c.blen) * mb + m0 / c.blen;
            if ready[block_index] > ex.now() {
                let stall = (ready[block_index] - ex.now()).as_picos();
                core.report.lock().unwrap().accumulator_dependency_stall_ps += stall;
                ex.resolve_at(ready[block_index]).await;
            }
            let mr = c.blen.min(m - m0);
            let nr = c.blen.min(n - spec.n);
            let kr = c.mlen.min(k - spec.k);
            for row in 0..mr {
                for col in 0..nr {
                    let target = (m0 + row) * n + spec.n + col;
                    for kk in 0..kr {
                        // Separate FP32 multiply/add, in ascending global K.
                        let product = input[(m0 + row) * k + spec.k + kk].to_f32()
                            * tile.values[col * c.mlen + kk].to_f32();
                        acc[target] += product;
                    }
                }
            }
            let service = service_cycles(a, c) * a.clock_period_ps;
            ready[block_index] =
                ex.now() + Duration::from_picos(service + extra_cycles(a, c) * a.clock_period_ps);
            {
                let mut report = core.report.lock().unwrap();
                report.useful_macs += (mr * nr * kr) as u64;
                report.issued_macs += (c.blen * c.blen * c.mlen) as u64;
                report.compute_busy_ps += service;
            }
            ex.resolve_at(Duration::from_picos(service)).await;
        }
        drop(tile);
        if issued < specs.len() {
            pending.push_back(spawn_load(
                core.clone(),
                shared.clone(),
                region.clone(),
                specs[issued],
            ));
            issued += 1;
        }
    }
    let last = ready.into_iter().max().unwrap();
    if last > ex.now() {
        core.report.lock().unwrap().pipeline_drain_ps += (last - ex.now()).as_picos();
        ex.resolve_at(last).await;
    }
    for (value, sum) in output.iter_mut().zip(acc.iter().copied()) {
        *value = bf16::from_f32(sum);
        if !value.is_finite() {
            return Err("GEMM produced non-finite BF16".into());
        }
    }
    Ok(())
}

async fn run_core(
    core: Arc<CoreState>,
    queue: Vec<Job>,
    w: Arc<Workload>,
    a: Arc<Architecture>,
    shared: Arc<Shared>,
) -> Result<(), String> {
    let mut private = queue.into_iter();
    loop {
        let job = {
            let _dispatch = shared.dispatcher.acquire().await.unwrap();
            let selected = if a.dispatch_policy == DispatchPolicy::Threshold {
                private.next()
            } else {
                let mut ready = shared.ready.lock().unwrap();
                // Prefer the core's own M class, then take any fitting group.
                // A free large core can drain remaining small jobs (and vice versa).
                let core_index = a.cores.iter().position(|c| c.id == core.config.id).unwrap();
                let index = ready
                    .iter()
                    .enumerate()
                    .filter(|(_, j)| job_fits(&w, &a, &core.config, j))
                    .min_by_key(|(_, j)| {
                        let preferred = if j.rows.len() >= a.dispatch_threshold {
                            a.large_core
                        } else {
                            a.small_core
                        };
                        (preferred != core_index, j.id)
                    })
                    .map(|(index, _)| index);
                index.map(|index| ready.remove(index))
            };
            if selected.is_some() {
                let duration = a.dispatch_cycles * a.clock_period_ps;
                shared.dispatcher_busy.fetch_add(duration, Ordering::SeqCst);
                Executor::current()
                    .resolve_at(Duration::from_picos(duration))
                    .await;
            }
            selected
        };
        let Some(job) = job else {
            break;
        };
        let start_ps = Executor::current().now().as_picos();
        let m = job.rows.len();
        let d = w.input_dim;
        let e = w.expert_hidden_dim;
        {
            let mut report = core.report.lock().unwrap();
            report.jobs += 1;
            report.vector_sram_peak_bytes = report.vector_sram_peak_bytes.max(vector_bytes(&w, m)?);
            report.accumulator_peak_bytes =
                report
                    .accumulator_peak_bytes
                    .max(accumulator_bytes(&w, m, &a, &core.config)?);
        }
        // These five BF16 regions are explicitly reserved, and retain their
        // contents until the output copy completes. No cross-core aliases.
        let mut x = vec![bf16::ZERO; m * d];
        let mut gate = vec![bf16::ZERO; m * e];
        let mut up = vec![bf16::ZERO; m * e];
        let mut z = vec![bf16::ZERO; m * e];
        let mut output = vec![bf16::ZERO; m * d];
        let mut accumulator = vec![0f32; m * d.max(e)];
        let wait = shared.vector.work(m * d).await;
        core.report.lock().unwrap().vector_wait_ps += wait;
        for (row, reference) in job.rows.iter().enumerate() {
            for col in 0..d {
                x[row * d + col] = bf16::from_bits(w.inputs_bf16[reference.token][col]);
            }
        }
        let expert = w.experts.iter().find(|x| x.id == job.expert).unwrap();
        gemm(
            &x,
            &mut gate,
            &mut accumulator,
            m,
            &expert.gate,
            &core,
            &a,
            &shared,
        )
        .await?;
        gemm(
            &x,
            &mut up,
            &mut accumulator,
            m,
            &expert.up,
            &core,
            &a,
            &shared,
        )
        .await?;
        let wait = shared.vector.work(m * e).await;
        core.report.lock().unwrap().vector_wait_ps += wait;
        for index in 0..m * e {
            let g = gate[index].to_f32();
            z[index] = bf16::from_f32((g / (1.0 + (-g).exp())) * up[index].to_f32());
            if !z[index].is_finite() {
                return Err("SwiGLU produced non-finite BF16".into());
            }
        }
        gemm(
            &z,
            &mut output,
            &mut accumulator,
            m,
            &expert.down,
            &core,
            &a,
            &shared,
        )
        .await?;
        let compute_done_ps = Executor::current().now().as_picos();
        let wait = shared.vector.work(m * d).await;
        core.report.lock().unwrap().vector_wait_ps += wait;
        {
            let mut reorder = shared.reorder.lock().unwrap();
            for (row, reference) in job.rows.iter().enumerate() {
                reorder[reference.output * d..(reference.output + 1) * d]
                    .copy_from_slice(&output[row * d..(row + 1) * d]);
            }
        }
        shared.completions.lock().unwrap().push(JobCompletion {
            job: job.id,
            expert: job.expert,
            shared: job.shared,
            core: core.config.id.clone(),
            rows: m,
            start_ps,
            compute_done_ps,
            output_copied_ps: Executor::current().now().as_picos(),
        });
        // All per-job buffers are dropped here, before the next job is admitted.
    }
    Ok(())
}

/// Execute inside an already-entered runtime::Executor. The public `run`
/// convenience wrapper owns one executor; callers integrating Ramulator may
/// instead spawn this future on their existing executor.
pub async fn execute(
    w: Workload,
    a: Architecture,
    hbm: Arc<dyn ErasedMemoryModel>,
    hbm_len: u64,
) -> Result<RunReport, String> {
    validate(&w, &a, hbm_len)?;
    if a.dma.is_some() && !hbm.supports_sector_reads() {
        return Err(
            "DMA sector/coalescing mode requires an explicitly capable memory backend".into(),
        );
    }
    let plan = plan(&w, &a)?;
    let ex = Executor::current();
    let begin = ex.now().as_picos();
    let w = Arc::new(w);
    let a = Arc::new(a);
    let queue_peak = plan.queues.iter().map(Vec::len).sum::<usize>() * 64;
    let shared = Arc::new(Shared {
        ready: Mutex::new(if a.dispatch_policy == DispatchPolicy::WorkConserving {
            plan.queues.iter().flatten().cloned().collect()
        } else {
            Vec::new()
        }),
        dispatcher: Semaphore::new(1),
        dispatcher_busy: AtomicU64::new(0),
        dma: Arc::new(Dma {
            hbm,
            credits: Semaphore::new(a.global_dma_credits),
            fair_pool: a
                .dma
                .as_ref()
                .filter(|d| d.fair_credits)
                .map(|_| Arc::new(CreditPool::new(a.global_dma_credits, a.cores.len()))),
            core_indices: a
                .cores
                .iter()
                .enumerate()
                .map(|(i, c)| (c.id.clone(), i))
                .collect(),
            inflight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            bytes: AtomicU64::new(0),
            config: a.dma.clone(),
            clock: a.clock_period_ps,
            lookup: (0..4).map(|_| Semaphore::new(1)).collect(),
            copy: (0..8).map(|_| Semaphore::new(1)).collect(),
            lines: Mutex::new(BTreeMap::new()),
            report: Mutex::new(DmaReport {
                fair_credit_reserve_per_core: if a.dma.as_ref().is_some_and(|d| d.fair_credits) {
                    a.global_dma_credits / 8
                } else {
                    0
                },
                reserved_bytes: if a.dma.is_some() {
                    frontend_bytes(&a)?
                } else {
                    0
                },
                ..Default::default()
            }),
        }),
        vector: Arc::new(VectorUnit {
            permit: Semaphore::new(1),
            lanes: a.vector_elements_per_cycle,
            clock: a.clock_period_ps,
            busy_ps: AtomicU64::new(0),
        }),
        reorder: Mutex::new(vec![bf16::ZERO; plan.output_rows * w.input_dim]),
        completions: Mutex::new(Vec::new()),
    });
    let cores: Vec<Arc<CoreState>> = a
        .cores
        .iter()
        .map(|c| Arc::new(CoreState::new(c, &a)))
        .collect();
    let mut pending = Vec::new();
    for (core, queue) in cores.iter().cloned().zip(plan.queues) {
        let w = w.clone();
        let a = a.clone();
        let shared = shared.clone();
        let (tx, rx) = oneshot::channel();
        pending.push(rx);
        ex.spawn(async move {
            let _ = tx.send(run_core(core, queue, w, a, shared).await);
        });
    }
    // All queues drain even if one returns an error; no orphaned tasks or credits.
    let mut failure = None;
    for rx in pending {
        match rx.await {
            Ok(Ok(())) => (),
            Ok(Err(e)) => {
                failure.get_or_insert(e);
            }
            Err(_) => {
                failure.get_or_insert("core task terminated without result".into());
            }
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    let d = w.input_dim;
    let tokens = w.inputs_bf16.len();
    let mut sums = vec![vec![0f32; d]; tokens];
    // One shared unit, one deterministic accumulation stream. A per-route
    // BF16 result is retained in bounded reorder SRAM until this point.
    shared.vector.work(plan.output_rows * d).await;
    {
        let reorder = shared.reorder.lock().unwrap();
        for (index, route) in plan.routes.iter().enumerate() {
            for col in 0..d {
                sums[route.token][col] += route.weight * reorder[index * d + col].to_f32();
            }
        }
        if let Some(s) = &w.shared_expert {
            for token in 0..tokens {
                for col in 0..d {
                    sums[token][col] +=
                        s.weight * reorder[(plan.shared_base + token) * d + col].to_f32();
                }
            }
        }
    }
    shared.vector.work(tokens * d).await;
    let output_bf16: Vec<Vec<u16>> = sums
        .iter()
        .map(|row| row.iter().map(|v| bf16::from_f32(*v).to_bits()).collect())
        .collect();
    let output_f32: Vec<Vec<f32>> = output_bf16
        .iter()
        .map(|row| row.iter().map(|v| bf16::from_bits(*v).to_f32()).collect())
        .collect();
    if output_f32.iter().flatten().any(|v| !v.is_finite()) {
        return Err("combined output is non-finite BF16".into());
    }
    let total_ps = ex.now().as_picos() - begin;
    let reports: Vec<CoreReport> = cores
        .iter()
        .map(|c| {
            let mut r = c.report.lock().unwrap().clone();
            r.weight_slots_peak = c.slots_peak.load(Ordering::SeqCst);
            r.weight_sram_peak_bytes = c.weight_peak.load(Ordering::SeqCst);
            r.cache_requests = c.cache.requests.load(Ordering::SeqCst);
            r.cache_hits = c.cache.hits.load(Ordering::SeqCst);
            r.cache_port_busy_ps = c.cache.busy.load(Ordering::SeqCst);
            r.cache_peak_bytes = c.cache.peak.load(Ordering::SeqCst);
            if total_ps > 0 {
                r.compute_busy_fraction = r.compute_busy_ps as f64 / total_ps as f64;
                r.mac_utilization = r.useful_macs as f64 * a.clock_period_ps as f64
                    / (r.multipliers as f64 * total_ps as f64);
            }
            r
        })
        .collect();
    let inflight_peak = shared.dma.peak.load(Ordering::SeqCst);
    let mut completions = shared.completions.lock().unwrap().clone();
    for j in &mut completions {
        j.start_ps -= begin;
        j.compute_done_ps -= begin;
        j.output_copied_ps -= begin;
    }
    Ok(RunReport {
        schema_version: 1, workload: w.name.clone(), architecture: a.name.clone(),
        timing_model: format!("{:?}: pipelined max(BLEN, ceil(BLEN*MLEN/activation_supply)) service with log2(MLEN)+pipeline readiness, or legacy serialized MLEN+overhead per instruction; shared ErasedMemoryModel; not RTL calibrated", a.matrix_timing),
        timing_boundary: "ready BF16 inputs/routes and resident MX weights -> core gathers, weight reads/decode, three numerical GEMMs, SwiGLU, route reorder, deterministic weighted combine -> ready BF16 output; excludes router, initial input/weight placement, output HBM store".into(),
        weight_format: "PLENA local E4M3/E8M0 block8, output-major [N,K], separate element/scale streams, decoded BF16 normal SRAM; not an OCP MX conformance claim".into(),
        total_ps, multipliers: reports.iter().map(|r| r.multipliers).sum(),
        useful_macs: reports.iter().map(|r| r.useful_macs).sum(), issued_macs: reports.iter().map(|r| r.issued_macs).sum(),
        hbm_read_bytes: shared.dma.bytes.load(Ordering::SeqCst), hbm_write_bytes: 0,
        global_dma_inflight_peak: inflight_peak, global_dma_staging_peak_bytes: inflight_peak * 64,
        combine_sram_peak_bytes: plan.global_storage, shared_vector_busy_ps: shared.vector.busy_ps.load(Ordering::SeqCst),
        dispatch_queue_peak_bytes: queue_peak, dispatcher_busy_ps: shared.dispatcher_busy.load(Ordering::SeqCst),
        dma_frontend: a.dma.as_ref().map(|_| shared.dma.report.lock().unwrap().clone()),
        cores: reports, job_completions: completions, output_bf16, output_f32, pre_round_output_f32: sums,
    })
}

/// Run a fresh deterministic simulation. HBM backing and timing are provided by
/// the caller and shared by all cores. The HBM image length is mandatory.
pub async fn run(
    w: Workload,
    a: Architecture,
    hbm: Arc<dyn ErasedMemoryModel>,
    hbm_len: u64,
) -> Result<RunReport, String> {
    validate(&w, &a, hbm_len)?;
    let ex = Executor::new();
    let (tx, mut rx) = oneshot::channel();
    ex.spawn(async move {
        let _ = tx.send(execute(w, a, hbm, hbm_len).await);
    });
    ex.enter(Instant::ETERNITY).await;
    rx.try_recv().map_err(|_| {
        "simulation stopped before completion (deadlock or task failure)".to_string()
    })?
}

#[cfg(test)]
mod dma_lifetime_tests {
    use super::*;

    #[tokio::test]
    async fn late_join_fetches_missing_sector_and_completed_entries_are_released() {
        let mut a = super::super::tests::architecture();
        a.dma = Some(DmaConfig {
            issue_policy: ramulator::model::IssuePolicy::PerChannel,
            sector_reads: true,
            coalesce: true,
            fair_credits: false,
            lookup_ii_cycles: 2,
            frontend_sram_bytes: 45056,
        });
        let backing = memory::MemoryBacked::with_capacity(64);
        backing.with_data(|bytes| {
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = i as u8;
            }
        });
        let native = ramulator::Ramulator::hbm2_preset(8).unwrap();
        let hbm = Arc::new(memory::WithStats::new(memory::WithTiming::new(
            native, backing,
        )));
        let dma = Arc::new(Dma {
            hbm: hbm.clone(),
            credits: Semaphore::new(2),
            fair_pool: None,
            core_indices: BTreeMap::new(),
            inflight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            bytes: AtomicU64::new(0),
            config: a.dma.clone(),
            clock: 1000,
            lookup: (0..4).map(|_| Semaphore::new(1)).collect(),
            copy: (0..8).map(|_| Semaphore::new(1)).collect(),
            lines: Mutex::new(BTreeMap::new()),
            report: Mutex::new(DmaReport::default()),
        });
        let core = Arc::new(CoreState::new(&a.cores[0], &a));
        let ex = Executor::new();
        let d = dma.clone();
        ex.spawn(async move {
            let first = d.read(0, 1, 32, core.clone());
            let late = async {
                Executor::current()
                    .resolve_at(Duration::from_picos(3000))
                    .await;
                d.read(0, 3, 64, core.clone()).await
            };
            let (a, b) = futures::join!(first, late);
            assert_eq!(&a[..32], &(0..32u8).collect::<Vec<_>>());
            assert!(
                a[32..].iter().all(|b| *b == 0),
                "unrequested sector leaked into result"
            );
            assert_eq!(b, std::array::from_fn(|i| i as u8));
            assert!(
                d.lines.lock().unwrap().is_empty(),
                "MSHR retained after final copy"
            );
            d.read(0, 1, 32, core).await;
        });
        ex.enter(Instant::ETERNITY).await;
        let r = dma.report.lock().unwrap();
        assert_eq!(
            (
                r.line_requests,
                r.sector_requests,
                r.merged_sectors,
                r.mshr_peak
            ),
            (3, 4, 1, 1)
        );
        assert_eq!(hbm.statistics().total_bytes_read, 96);
        assert_eq!(dma.bytes.load(Ordering::SeqCst), 96);
    }
}
