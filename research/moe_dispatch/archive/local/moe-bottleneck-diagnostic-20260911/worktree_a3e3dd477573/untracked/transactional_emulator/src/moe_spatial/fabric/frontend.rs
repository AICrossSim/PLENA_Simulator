//! Bounded output-band admission and independently reserved weight prefetch.
//! Estimates use dimensions, configured rates and past completions only.
use super::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrontendConfig {
    pub whole_bands: bool,
    pub prefetch: bool,
    pub band_window: usize,
    pub initial_tile_latency: u64,
    pub aging_latency_multiple: u64,
    pub control_storage_bytes: usize,
}
impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            whole_bands: false,
            prefetch: false,
            band_window: 32,
            initial_tile_latency: 500,
            aging_latency_multiple: 4,
            control_storage_bytes: 4096,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct FrontendReport {
    pub reserved_control_bytes: usize,
    pub required_control_bytes: usize,
    pub reserved_per_core: Vec<usize>,
    pub band_assignments: Vec<usize>,
    pub completed_bands: usize,
    pub band_queue_peak: Vec<usize>,
    pub bands_inflight_peak: usize,
    pub selection_service_cycles: u64,
    pub assignment_service_cycles: u64,
    pub candidate_checks: u64,
    pub prefetch_descriptors: u64,
    pub prefetch_peak: Vec<usize>,
    pub no_operand_stage_prefetches: u64,
    pub source_latency_samples: u64,
    pub source_latency_ewma: u64,
    pub weight_requests_by_core: Vec<u64>,
    pub whole_band_ownership_verified: bool,
}
struct Band {
    core: usize,
    ready: u64,
    rows_left: usize,
    estimate: u64,
}
pub(super) struct Frontend {
    pub config: FrontendConfig,
    pub report: FrontendReport,
    bands: BTreeMap<(usize, usize), Band>,
    next_job: usize,
    next_nt: usize,
    work: Vec<u64>,
    queue: Vec<usize>,
    pub select_ready: u64,
    pub assign_ready: u64,
    pub cursor: usize,
    pub last_request: Vec<u64>,
}
impl Frontend {
    pub fn new(
        cfg: FrontendConfig,
        r: &Request,
        f: &Fabric,
        p: &mut PrivateState,
    ) -> Result<Self, String> {
        if cfg.band_window == 0
            || cfg.band_window > f.ready_window
            || cfg.initial_tile_latency == 0
            || cfg.aging_latency_multiple == 0
            || f.control != Control::TileCohort
            || r.ownership != Ownership::TileStealing
        {
            return Err("frontend requires bounded window, private tile-cohort output affinity and nonzero latency/aging".into());
        }
        let nc = r.m_lanes.len();
        // 64 B active band, 32 B slot/DMA tag, 96 B/core counters + 64 B global.
        let required =
            cfg.band_window * 64 + p.config.weight_slots.iter().sum::<usize>() * 32 + nc * 96 + 64;
        if required > cfg.control_storage_bytes {
            return Err("frontend control SRAM budget exceeded".into());
        }
        let total_m = r.m_lanes.iter().sum::<usize>();
        let mut reserved: Vec<_> = r
            .m_lanes
            .iter()
            .map(|m| cfg.control_storage_bytes * m / total_m / 32 * 32)
            .collect();
        reserved[0] += cfg.control_storage_bytes - reserved.iter().sum::<usize>();
        for (c, &n) in reserved.iter().enumerate() {
            if p.report.accumulator_peak_bytes[c] + n > p.config.accumulator_bytes[c] {
                return Err("no room for frontend control records".into());
            }
            // Same accumulator budget as values and old 16-byte output records.
            p.report.accumulator_peak_bytes[c] += n;
        }
        if r.jobs
            .iter()
            .map(|j| j.m * j.n.div_ceil(r.n_lanes) * (4 * r.n_lanes + 16))
            .sum::<usize>()
            + cfg.control_storage_bytes
            > f.accumulator_bytes
        {
            return Err("outputs plus frontend exceed total accumulator SRAM".into());
        }
        Ok(Self {
            report: FrontendReport {
                reserved_control_bytes: cfg.control_storage_bytes,
                required_control_bytes: required,
                reserved_per_core: reserved,
                band_assignments: vec![0; nc],
                band_queue_peak: vec![0; nc],
                prefetch_peak: vec![0; nc],
                source_latency_ewma: cfg.initial_tile_latency,
                weight_requests_by_core: vec![0; nc],
                ..Default::default()
            },
            config: cfg,
            bands: BTreeMap::new(),
            next_job: 0,
            next_nt: 0,
            work: vec![0; nc],
            queue: vec![0; nc],
            select_ready: 0,
            assign_ready: 0,
            cursor: 0,
            last_request: vec![0; nc],
        })
    }
    pub fn allowed(&self, key: Key, c: usize, now: u64) -> bool {
        !self.config.whole_bands
            || self
                .bands
                .get(&(key.job, key.nt))
                .is_some_and(|b| b.core == c && b.ready <= now)
    }
    pub fn candidate_keys(
        &self,
        c: usize,
        now: u64,
        ready: &Ready,
        slots: &[Vec<Slot>],
        limit: usize,
    ) -> Vec<Key> {
        let mut result = Vec::new();
        let bands: Vec<_> = if self.config.whole_bands {
            self.bands
                .iter()
                .filter(|(_, b)| b.core == c && b.ready <= now)
                .map(|(&key, _)| key)
                .collect()
        } else {
            slots[c]
                .iter()
                .filter_map(|s| s.key.map(|k| (k.job, k.nt)))
                .collect()
        };
        // Each resident band has a next-ready-K pointer/bitmap updated at
        // commit. This host range lookup represents that bounded index, not
        // scanning the global pending list past thousands of unopened bands.
        for (job, nt) in bands {
            if let Some((&key, _)) = ready
                .groups
                .range(
                    Key { job, nt, kt: 0 }..=Key {
                        job,
                        nt,
                        kt: usize::MAX,
                    },
                )
                .next()
                && !result.contains(&key)
            {
                result.push(key);
            }
        }
        if !self.config.whole_bands {
            for &key in ready.fifo.iter().take(limit) {
                if result.len() == limit {
                    break;
                }
                if !result.contains(&key) {
                    result.push(key);
                }
            }
        }
        result.truncate(limit);
        result
    }
    pub fn source_completed(&mut self, latency: u64) {
        self.report.source_latency_samples += 1;
        self.report.source_latency_ewma = (7 * self.report.source_latency_ewma + latency)
            .div_ceil(8)
            .max(1);
    }
    pub fn has_pending_assignment(&self, r: &Request) -> bool {
        self.config.whole_bands && self.next_job < r.jobs.len()
    }
    pub fn admit(
        &mut self,
        now: u64,
        input: &Input,
        p: &mut PrivateState,
        slots: &[Vec<Slot>],
        cp: &mut Port,
        log: &mut Vec<Service>,
    ) {
        let (r, f) = (&input.compute, &input.fabric);
        if !self.has_pending_assignment(r)
            || now < self.assign_ready
            || self.bands.len() >= self.config.band_window
        {
            return;
        }
        let j = &r.jobs[self.next_job];
        let key = Key {
            job: self.next_job,
            nt: self.next_nt,
            kt: 0,
        };
        let bytes = j.m * (4 * r.n_lanes + 16);
        let nc = r.m_lanes.len();
        let best = (0..nc)
            .filter(|&c| {
                self.queue[c] < self.config.band_window.div_ceil(nc)
                    && p.report.accumulator_peak_bytes[c] + bytes <= p.config.accumulator_bytes[c]
            })
            .map(|c| {
                let m = j.m.div_ceil(r.m_lanes[c]) as u64;
                let k = j.k.div_ceil(r.k_lanes) as u64;
                let issue = r
                    .issue_interval_cycles
                    .max(service(
                        aligned(r.n_lanes * r.k_lanes * 2),
                        p.config.weight_read_bpc[c],
                        false,
                    ))
                    .max(service(
                        aligned(r.m_lanes[c] * r.k_lanes * 2),
                        p.config.activation_read_bpc[c],
                        false,
                    ))
                    .max(service(
                        aligned(r.m_lanes[c] * r.n_lanes * 8),
                        p.config.accumulator_bpc[c],
                        false,
                    ));
                // Queued work overlaps pipeline tails across independent bands.
                let work = k * m * issue;
                let port = p.weight_write[c].free[0]
                    .max(p.activation_write[c].free[0])
                    .max(p.accumulator[c].free[0]);
                let pending = slots[c].iter().filter(|s| s.ready > now).count() as u64;
                let supply = now
                    + self.report.source_latency_ewma
                    + pending * service((r.n_lanes * r.k_lanes * 2) as u64, f.weight_bpc, false);
                let dependency_span = (k - 1) * (r.result_latency_cycles + 1).max(m * issue)
                    + m * issue
                    + r.result_latency_cycles;
                let start = (now + self.work[c]).max(supply).max(port);
                let finish = (start + work + r.result_latency_cycles).max(now + dependency_span);
                let tail = (m as usize * r.m_lanes[c] - j.m) * j.n.min(r.n_lanes) * j.k;
                ((finish, tail, (c + nc - self.cursor) % nc), c, work)
            })
            .min_by_key(|x| x.0);
        let Some((_, c, work)) = best else {
            return;
        };
        // Comparisons and row-bit/descriptor installation are a charged service.
        let cycles = control(nc as u64 + 1 + j.m.div_ceil(8) as u64, f);
        let end = cp.reserve(now, cycles, 0, &[], log);
        self.report.assignment_service_cycles += cycles;
        self.assign_ready = end.max(now + 1);
        for row in 0..j.m {
            p.claim(r, key, row, c);
        }
        self.bands.insert(
            (key.job, key.nt),
            Band {
                core: c,
                ready: end,
                rows_left: j.m,
                estimate: work,
            },
        );
        self.work[c] += work;
        self.queue[c] += 1;
        self.report.band_assignments[c] += 1;
        self.report.band_queue_peak[c] = self.report.band_queue_peak[c].max(self.queue[c]);
        self.report.bands_inflight_peak = self.report.bands_inflight_peak.max(self.bands.len());
        self.cursor = (c + 1) % nc;
        self.next_nt += 1;
        if self.next_nt == j.n.div_ceil(r.n_lanes) {
            self.next_job += 1;
            self.next_nt = 0;
        }
    }
    pub fn complete_row(&mut self, key: Key) {
        if !self.config.whole_bands {
            return;
        }
        let b = self.bands.get_mut(&(key.job, key.nt)).unwrap();
        b.rows_left -= 1;
        if b.rows_left == 0 {
            let b = self.bands.remove(&(key.job, key.nt)).unwrap();
            self.queue[b.core] -= 1;
            self.work[b.core] -= b.estimate;
            self.report.completed_bands += 1;
        }
    }
    pub fn audit(&mut self, r: &Request, p: &PrivateState) {
        if self.config.whole_bands {
            assert!(self.bands.is_empty() && self.next_job == r.jobs.len());
            for (ji, j) in r.jobs.iter().enumerate() {
                for nt in 0..j.n.div_ceil(r.n_lanes) {
                    let c = p.owners[&(ji, nt, 0)];
                    assert!((0..j.m).all(|row| p.owners[&(ji, nt, row)] == c));
                }
            }
            self.report.whole_band_ownership_verified = true;
        }
    }
}
