//! Residual-capacity admission. Estimates affect admission/priority only;
//! ownership, SRAM reservation, DMA handshake and arithmetic dependencies win.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Thresholds {
    pub depth: usize,
    pub low: u64,
    pub target: u64,
}

pub(super) struct PhaseAhead {
    pub phase: usize,
    pub tile: Tile,
}

#[derive(Serialize)]
pub(super) struct Diagnostics {
    #[serde(skip)]
    pub accepted_this_cycle: Vec<bool>,
    #[serde(skip)]
    pub front_wait_code: Vec<usize>,
    #[serde(skip)]
    accepted_at: BTreeMap<u64, u64>,
    pub credit_lease_cycles_sum: u64,
    pub credit_lease_cycles_max: u64,
    pub credit_completions: u64,
    pub cores: Vec<CoreSupply>,
}
#[derive(Serialize, Default)]
pub(super) struct CoreSupply {
    pub observed_cycles: u64,
    pub no_accept_cycles: u64,
    pub idle_reason_and_other_phase: BTreeMap<usize, u64>,
    pub slot_occupancy_cycles: BTreeMap<usize, u64>,
    pub tier_grants: [u64; 4],
    pub tier0_eligible_cycles: u64,
    pub tier0_episodes: u64,
    #[serde(skip)]
    was_tier0: bool,
    pub phase_ahead_tiles: u64,
    pub next_denied_for_current_depth: u64,
}
impl Diagnostics {
    pub fn new(n: usize) -> Self {
        Self {
            accepted_this_cycle: vec![false; n],
            front_wait_code: vec![0; n],
            accepted_at: BTreeMap::new(),
            credit_lease_cycles_sum: 0,
            credit_lease_cycles_max: 0,
            credit_completions: 0,
            cores: (0..n).map(|_| CoreSupply::default()).collect(),
        }
    }
}

impl Sim {
    pub(super) fn load_surplus_thresholds(&mut self, c: usize) {
        if self.cfg.surplus_rules == 0 {
            return;
        }
        let me = self.w.experts[self.cores[c].session.as_ref().unwrap().e].m;
        let index = ((usize::BITS - (me - 1).leading_zeros()) as usize).min(7);
        let entry = &self.w.engine_layout["surplus_lut"]["cores"][c][index];
        self.cores[c].surplus_thresholds = Some(Thresholds {
            depth: entry["depth"].as_u64().unwrap() as usize,
            low: entry["low_bytes"].as_u64().unwrap(),
            target: entry["target_bytes"].as_u64().unwrap(),
        });
    }
    /// Inventory counts accepted bytes, both in flight and landed, exactly once.
    /// The tile is retained until its last operand read; no landing increment.
    pub(super) fn current_inventory(&self, c: usize) -> u64 {
        let core = &self.cores[c];
        let Some(r) = &core.run else {
            return core.incoming.as_ref().map_or(0, |t| t.sent as u64);
        };
        core.live_tiles
            .iter()
            .flatten()
            .filter_map(|&t| r.tiles.get(t))
            .filter(|t| !t.retired)
            .map(|t| t.sent as u64)
            .sum()
    }
    fn current_held_slots(&self, c: usize) -> usize {
        let core = &self.cores[c];
        core.run
            .as_ref()
            .map_or(usize::from(core.incoming.is_some()), |r| {
                core.live_tiles
                    .iter()
                    .flatten()
                    .filter(|&&t| t < r.tiles.len())
                    .count()
            })
    }
    pub(super) fn borrowable_slots(&self, c: usize) -> usize {
        let core = &self.cores[c];
        let Some(entry) = core.surplus_thresholds.as_ref() else {
            return core.free_slots.len();
        };
        let held = self.current_held_slots(c);
        // Near a stage boundary, never protect more slots than actual remaining
        // active-stage tiles. Future stage weights have separate ownership tags.
        let remaining = core.run.as_ref().map_or_else(
            || {
                core.session.as_ref().map_or(0, |s| {
                    if s.phase <= 1 {
                        let e = &self.w.experts[s.e];
                        ceil(e.f, 4) * ceil(e.h, 512)
                    } else {
                        0
                    }
                })
            },
            |r| r.tiles.len() - r.admit + held,
        );
        let depth = entry.depth.min(remaining);
        core.free_slots
            .len()
            .saturating_sub(depth.saturating_sub(held))
    }
    pub(super) fn surplus_due(&self, c: usize, e: usize) -> bool {
        self.cores[c].session.is_none()
            || self.current_requests_sent(c)
            || self.progress_estimate(c)
                <= self.first_ready_estimate(e, c) + self.cfg.surplus_margin_cycles
    }
    pub(super) fn request_tier(&self, req: &DmaRequest) -> (usize, u64) {
        let c = req.core;
        let core = &self.cores[c];
        let inventory = self.current_inventory(c);
        if core.next.as_ref().is_some_and(|n| n.e == req.task) {
            return (if self.surplus_due(c, req.task) { 1 } else { 3 }, inventory);
        }
        if core.ahead.as_ref().is_some_and(|a| a.phase == req.phase) {
            return (3, inventory);
        }
        let Some(entry) = core.surplus_thresholds.as_ref() else {
            return (0, inventory);
        };
        let low = entry.low;
        let target = entry.target;
        (
            if inventory < low {
                0
            } else if inventory < target {
                2
            } else {
                3
            },
            inventory,
        )
    }
    pub(super) fn admit_phase_ahead(&mut self) {
        if self.cfg.surplus_rules < 4 || self.now < self.control_free {
            return;
        }
        for off in 0..self.cores.len() {
            let c = (self.rr_desc + off) % self.cores.len();
            let Some(s) = &self.cores[c].session else {
                continue;
            };
            let phase = match s.phase {
                0 => 1,
                1 | 2 => 4,
                _ => continue,
            };
            if self.cores[c].ahead.is_some() || self.borrowable_slots(c) == 0 {
                continue;
            }
            let e = s.e;
            let x = &self.w.experts[e];
            let (n, k) = if phase == 1 { (x.f, x.h) } else { (x.h, x.f) };
            let nv = n.min(4);
            let kv = k.min(512);
            let bytes = nv * align(kv * 2, 32);
            let slot = self.cores[c].free_slots.pop().unwrap();
            let cost = if self.cfg.control_cost { 2 } else { 0 };
            self.control_free = self.now + cost;
            self.cores[c].live_tiles[slot] = Some(usize::MAX - 1);
            self.cores[c].ahead = Some(PhaseAhead {
                phase,
                tile: Tile {
                    n_start: 0,
                    k_start: 0,
                    nv,
                    kv,
                    slot: Some(slot),
                    release: self.control_free,
                    sent: 0,
                    acks: 0,
                    bytes,
                    ready: false,
                    retired: false,
                },
            });
            self.cores[c].stats.control_cycles += cost;
            self.cores[c].stats.weight_bytes += bytes as u64;
            let used = self.cores[c].wslots - self.cores[c].free_slots.len();
            self.cores[c].stats.weight_peak_bytes =
                self.cores[c].stats.weight_peak_bytes.max(used * 4096);
            self.diagnostics.cores[c].phase_ahead_tiles += 1;
            self.log(
                json!({"event":"phase_ahead_reserved","cycle":self.now,"core":c,
                "task":e,"phase":phase,"slot":slot,"bytes":bytes}),
            );
            self.rr_desc = (c + 1) % self.cores.len();
            self.last_progress = self.now;
            break;
        }
    }
    pub(super) fn record_dma_accept(&mut self, r: &DmaRequest) {
        self.diagnostics.accepted_this_cycle[r.core] = true;
        assert!(
            self.diagnostics
                .accepted_at
                .insert(r.serial, self.now)
                .is_none()
        );
        if self.cfg.surplus_rules >= 3 {
            let tier = self.request_tier(r).0;
            self.diagnostics.cores[r.core].tier_grants[tier] += 1;
        }
    }
    pub(super) fn record_dma_landing(&mut self, r: &DmaRequest) {
        let accepted = self.diagnostics.accepted_at.remove(&r.serial).unwrap();
        let lease = self.now - accepted;
        self.diagnostics.credit_lease_cycles_sum += lease;
        self.diagnostics.credit_lease_cycles_max =
            self.diagnostics.credit_lease_cycles_max.max(lease);
        self.diagnostics.credit_completions += 1;
    }
    pub(super) fn observe_supply(&mut self) {
        for c in 0..self.cores.len() {
            let core = &self.cores[c];
            let current = self.current_held_slots(c);
            let next = usize::from(core.next.as_ref().is_some_and(|n| n.first.is_some()));
            let ahead = usize::from(core.ahead.is_some());
            assert_eq!(current + next + ahead + core.free_slots.len(), core.wslots);
            let occupancy = current * 1000 + next * 100 + ahead * 10 + core.free_slots.len();
            let candidate = self.candidate(c);
            let tier0 = self.cfg.surplus_rules >= 3
                && candidate
                    .as_ref()
                    .is_some_and(|r| self.request_tier(r).0 == 0);
            let other = if self.cores.len() == 1 {
                7
            } else {
                self.cores[1 - c].session.as_ref().map_or(6, |s| s.phase)
            };
            let other_wait = if self.cores.len() == 1 {
                0
            } else {
                self.diagnostics.front_wait_code[1 - c]
            };
            let reason = if candidate.is_none() {
                0
            } else if self.credit_used >= self.cfg.credits {
                1
            } else if self.now < self.cfg.dma_ready_after
                || self.now % self.cfg.dma_ready_period >= self.cfg.dma_ready_cycles
            {
                2
            } else {
                3
            };
            let d = &mut self.diagnostics.cores[c];
            d.observed_cycles += 1;
            *d.slot_occupancy_cycles.entry(occupancy).or_default() += 1;
            if !self.diagnostics.accepted_this_cycle[c] {
                d.no_accept_cycles += 1;
                *d.idle_reason_and_other_phase
                    .entry(reason * 1000 + other * 100 + other_wait)
                    .or_default() += 1;
            }
            if tier0 {
                d.tier0_eligible_cycles += 1;
                if !d.was_tier0 {
                    d.tier0_episodes += 1;
                }
            }
            d.was_tier0 = tier0;
        }
    }
}
