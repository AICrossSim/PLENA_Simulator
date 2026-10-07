//! Live native HBM weight timing. Values remain in the verified BF16 operand bank.
//! Each tile holds its finite SRAM reservation until all native sectors return.
use super::{
    Request,
    fabric::{WeightRead, WeightSource},
};
use ramulator::config::{
    self, AddrMapper, Controller, DDRController, DRAM, RefreshManager, RowPolicy, Scheduler,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    mem::ManuallyDrop,
    rc::Rc,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub channels: usize,
    pub core_period_ps: u64,
    pub outstanding_sectors: usize,
    /// Fixed logical expert ID to address mapping, independent of active experts.
    pub expert_stride_bytes: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            channels: 8,
            core_period_ps: 1000,
            outstanding_sectors: 256,
            expert_stride_bytes: 16 * 1024 * 1024,
        }
    }
}
struct Tile {
    // Host expansion of a 2D DMA descriptor, not an extra payload buffer.
    by_channel: Vec<VecDeque<u64>>,
    remaining: usize,
    released: u64,
}
#[derive(Default, Serialize)]
struct Counters {
    submitted_tiles: u64,
    completed_tiles: u64,
    accepted_sectors: u64,
    completed_sectors: u64,
    native_read_bytes: u64,
    useful_weight_bytes: u64,
    outstanding_peak: usize,
    active_tile_peak: usize,
    ingress_credit_blocked_cycles: u64,
    rejected_per_channel: Vec<u64>,
    accepted_per_channel: Vec<u64>,
    tile_latency_cycles_sum: u64,
}
#[derive(Serialize)]
struct Address {
    expert: usize,
    base: u64,
    row_stride: u64,
    n: usize,
    k: usize,
}
pub struct Native {
    // Process-owned, as in the existing native runner. Drain is asserted before exit.
    raw: ManuallyDrop<ramulator::raw::Ramulator>,
    config: Config,
    period_ps: u64,
    next_tick_ps: u64,
    last_cycle: Option<u64>,
    pending: usize,
    tiles: BTreeMap<usize, Tile>,
    callbacks: Rc<RefCell<Vec<(usize, u64)>>>,
    deferred_consumer: bool,
    consumer_pending: usize,
    sector_completions: Vec<(usize, u64)>,
    counts: Counters,
    addresses: Vec<Address>,
    library_path: String,
    library_sha256: String,
    accept_hash: Sha256,
    complete_hash: Sha256,
}
impl Native {
    pub fn new(config: Config, request: &Request) -> Result<Self, String> {
        if !config.channels.is_power_of_two()
            || config.channels > 32
            || config.core_period_ps == 0
            || config.core_period_ps > 100_000
            || config.outstanding_sectors == 0
            || config.outstanding_sectors > 4096
            || config.expert_stride_bytes == 0
            || !config.expert_stride_bytes.is_multiple_of(4096)
        {
            return Err("invalid native timing/credit/address configuration".into());
        }
        if !request.k_lanes.is_multiple_of(16) {
            return Err("native BF16 K tile width must be a multiple of 16".into());
        }
        // Sorted, dense banks would remap weights whenever routing changes. Keep IDs fixed.
        let mut addresses = Vec::new();
        for j in &request.jobs {
            let row_stride = (j.k as u64 * 2).div_ceil(32) * 32;
            if j.n as u64 * row_stride > config.expert_stride_bytes {
                return Err("expert matrix exceeds fixed address stride".into());
            }
            let base = (j.expert as u64)
                .checked_mul(config.expert_stride_bytes)
                .ok_or("native address overflow")?;
            // Capacity from native geometry: 2 pseudochannels * 16 banks * 65536 rows * 128 columns * 8 bytes.
            if base
                .checked_add(config.expert_stride_bytes)
                .ok_or("native address overflow")?
                > config.channels as u64 * 2 * 1024 * 1024 * 1024
            {
                return Err("expert address exceeds configured HBM capacity; remap non-model sentinel IDs explicitly".into());
            }
            addresses.push(Address {
                expert: j.expert,
                base,
                row_stride,
                n: j.n,
                k: j.k,
            });
        }
        let controller = Controller::HBM12(DDRController {
            options: Default::default(),
            scheduler: Scheduler::FrFcFs,
            refresh_manager: RefreshManager::all_bank(),
            row_policy: RowPolicy::Open,
            addr_mapper: AddrMapper::MOP4CLXOR,
            dram: DRAM::HBM2(config::hbm2::HBM2 {
                timing: config::hbm2::HBM2Timing::HBM2_2000MBPS,
                org: config::hbm2::HBM2Org::HBM2_8GB,
                diagnostic: config::hbm2::HbmDiagnostic::Native,
            }),
        });
        let mut raw = ramulator::raw::Ramulator::new(config::Config {
            controllers: vec![controller; config.channels],
            channel_mapper: Default::default(),
        })
        .map_err(|e| e.to_string())?;
        let period_ps = raw.period() as u64;
        let library_path = ramulator::raw::Ramulator::library_path();
        let library_sha256 = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&library_path).map_err(|e| e.to_string())?)
        );
        Ok(Self {
            raw: ManuallyDrop::new(raw),
            period_ps,
            next_tick_ps: period_ps,
            counts: Counters {
                accepted_per_channel: vec![0; config.channels],
                rejected_per_channel: vec![0; config.channels],
                ..Default::default()
            },
            config,
            last_cycle: None,
            pending: 0,
            tiles: BTreeMap::new(),
            callbacks: Default::default(),
            deferred_consumer: false,
            consumer_pending: 0,
            sector_completions: Vec::new(),
            addresses,
            library_path,
            library_sha256,
            accept_hash: Sha256::new(),
            complete_hash: Sha256::new(),
        })
    }
    pub fn report(&mut self) -> serde_json::Value {
        serde_json::json!({
            "scope":"live weight-only native HBM2 timing; X and accumulator on-chip; not complete layer/model",
            "config":self.config,"native_period_ps":self.period_ps,"native_transaction_bytes":32,
            "ingress":"one attempt/channel/core cycle; 256 default accepted sector credits; per-channel tile-order FIFO",
            "submission_boundary":"new tile first attempted at next core edge; completions sampled at core edges",
            "weight_representation":"uncompressed BF16 row-major W[N,K], each row padded to 32 B; no MX scale codec",
            "values":"SHA256-verified operand bank; Ramulator is a timing model, not a data store",
            "counts":self.counts,"pending":self.pending,"drained":self.drained(),"address_map":self.addresses,
            "consumer_pending":self.consumer_pending,"deferred_consumer":self.deferred_consumer,
            "native_library_path":self.library_path,"native_library_sha256":self.library_sha256,
            "acceptance_sha256":format!("{:x}",self.accept_hash.clone().finalize()),
            "completion_sha256":format!("{:x}",self.complete_hash.clone().finalize()),
            "native_stats":self.raw.native_stats(),
            "descriptor_implementation":"host expands 2D tile requests for timing; no synthesized frontend area claim"
        })
    }
    fn completions(&mut self, now: u64, done: &mut Vec<usize>) {
        for (id, address) in self.callbacks.take() {
            assert!(self.pending > 0);
            self.pending -= 1;
            if self.deferred_consumer {
                self.consumer_pending += 1;
                self.sector_completions.push((id, address));
            }
            self.counts.completed_sectors += 1;
            self.complete_hash.update(now.to_le_bytes());
            self.complete_hash.update((id as u64).to_le_bytes());
            let tile = self
                .tiles
                .get_mut(&id)
                .expect("callback must have live owner");
            tile.remaining -= 1;
            if tile.remaining == 0 {
                assert!(tile.by_channel.iter().all(|q| q.is_empty()));
                self.counts.tile_latency_cycles_sum += now - tile.released;
                self.counts.completed_tiles += 1;
                done.push(id);
            }
        }
        for id in done.iter() {
            self.tiles.remove(id);
        }
    }
    /// Retain each source credit until the bounded destination write acknowledges.
    /// Legacy callers retain their previous timing unless explicitly enabled.
    pub fn enable_consumer_backpressure(&mut self) {
        self.deferred_consumer = true;
    }
    pub fn take_sector_completions(&mut self) -> Vec<(usize, u64)> {
        std::mem::take(&mut self.sector_completions)
    }
    pub fn acknowledge_sector(&mut self) {
        assert!(self.deferred_consumer && self.consumer_pending > 0);
        self.consumer_pending -= 1;
    }
}
impl WeightSource for Native {
    fn submit(&mut self, request: WeightRead<'_>) -> Result<(), String> {
        let WeightRead {
            id,
            job,
            n_start,
            k_start,
            valid_n,
            valid_k,
            now,
        } = request;
        if valid_n == 0 || valid_k == 0 || !k_start.is_multiple_of(16) {
            return Err("native BF16 tile must start on a 32 B sector boundary".into());
        }
        let a = self
            .addresses
            .iter()
            .find(|a| a.expert == job.expert)
            .ok_or("missing native bank")?;
        let mut by_channel = vec![VecDeque::new(); self.config.channels];
        let sectors = valid_k.div_ceil(16);
        for n in n_start..n_start + valid_n {
            for s in 0..sectors {
                let addr = a.base + n as u64 * a.row_stride + k_start as u64 * 2 + s as u64 * 32;
                let ch = (addr / 32) as usize & (self.config.channels - 1);
                by_channel[ch].push_back(addr);
            }
        }
        let count = valid_n * sectors;
        if self
            .tiles
            .insert(
                id,
                Tile {
                    by_channel,
                    remaining: count,
                    released: now,
                },
            )
            .is_some()
        {
            return Err("duplicate native transfer ID".into());
        }
        self.counts.submitted_tiles += 1;
        self.counts.useful_weight_bytes += (valid_n * valid_k * 2) as u64;
        self.counts.active_tile_peak = self.counts.active_tile_peak.max(self.tiles.len());
        Ok(())
    }
    fn advance(&mut self, now: u64) -> Result<Vec<usize>, String> {
        if self.last_cycle.is_some_and(|n| now != n + 1) {
            return Err("native source requires consecutive core cycles".into());
        }
        self.last_cycle = Some(now);
        let ps = now
            .checked_mul(self.config.core_period_ps)
            .ok_or("native clock overflow")?;
        while self.next_tick_ps <= ps {
            self.raw.tick();
            self.next_tick_ps += self.period_ps;
        }
        let mut done = Vec::new();
        self.completions(now, &mut done);
        if self.pending + self.consumer_pending == self.config.outstanding_sectors
            && self
                .tiles
                .values()
                .any(|t| t.by_channel.iter().any(|q| !q.is_empty()))
        {
            self.counts.ingress_credit_blocked_cycles += 1;
        }
        for ch in 0..self.config.channels {
            if self.pending + self.consumer_pending >= self.config.outstanding_sectors {
                break;
            }
            let candidate = self
                .tiles
                .iter()
                .find_map(|(&id, t)| t.by_channel[ch].front().map(|&addr| (id, addr)));
            if let Some((id, addr)) = candidate {
                let callbacks = self.callbacks.clone();
                if self
                    .raw
                    .read(addr, move || callbacks.borrow_mut().push((id, addr)))
                {
                    self.tiles.get_mut(&id).unwrap().by_channel[ch].pop_front();
                    self.pending += 1;
                    self.counts.accepted_sectors += 1;
                    self.counts.accepted_per_channel[ch] += 1;
                    self.counts.native_read_bytes += 32;
                    self.accept_hash.update(now.to_le_bytes());
                    self.accept_hash.update(addr.to_le_bytes());
                    self.accept_hash.update((id as u64).to_le_bytes());
                    self.counts.outstanding_peak = self
                        .counts
                        .outstanding_peak
                        .max(self.pending + self.consumer_pending);
                } else {
                    self.counts.rejected_per_channel[ch] += 1;
                }
            }
        }
        // Some native implementations may complete accepted requests synchronously.
        self.completions(now, &mut done);
        Ok(done)
    }
    fn drained(&self) -> bool {
        self.pending == 0
            && self.consumer_pending == 0
            && self.sector_completions.is_empty()
            && self.tiles.is_empty()
            && self.callbacks.borrow().is_empty()
            && self.counts.accepted_sectors == self.counts.completed_sectors
            && self.counts.submitted_tiles == self.counts.completed_tiles
    }
}
