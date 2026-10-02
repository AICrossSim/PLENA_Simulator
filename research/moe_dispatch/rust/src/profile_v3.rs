//! M0 observation only. Every observer operation leaves modeled resources untouched.
use super::*;

#[derive(Default, Serialize)]
pub(super) struct Timeline {
    core: usize,
    task: usize,
    phase: usize,
    issue: usize,
    x_ready: u64,
    w_ready: u64,
    operand_start: u64,
    operand_end: u64,
    mac_start: u64,
    mac_end: u64,
    rmw_start: u64,
    accumulator_done: u64,
}
pub(super) struct Profile {
    h: [u64; 6],
    c: Vec<[u64; 9]>,
    returned_pending: usize,
    prev_dma: u64,
    prev_issue: Vec<Option<u64>>,
    intervals: Vec<BTreeMap<u64, u64>>,
    pending: BTreeMap<(usize, usize), Timeline>,
    samples: Vec<Timeline>,
    phases: BTreeMap<String, [u64; 4]>,
    ready: BTreeMap<(usize, usize, usize), u64>,
}
impl Profile {
    pub fn new(n: usize) -> Self {
        Self {
            h: [0; 6],
            c: vec![[0; 9]; n],
            returned_pending: 0,
            prev_dma: 0,
            prev_issue: vec![None; n],
            intervals: vec![BTreeMap::new(); n],
            pending: BTreeMap::new(),
            samples: vec![],
            phases: BTreeMap::new(),
            ready: BTreeMap::new(),
        }
    }
    pub fn report(&self, cycles: u64) -> Value {
        json!({"scope":"observer; no timing/resource changes", "hbm_states":
            {"H0":self.h[0],"H1a":self.h[1],"H1b":self.h[2],"H2":self.h[3],"H3":self.h[4],"H4":self.h[5]},
            "core_states":self.c.iter().map(|x|json!({"C0":x[0],"C1":x[1],"C2":x[2],"C3":x[3],"C4":x[4],"C5":x[5],"C6":x[6],"C7":x[7],"C8":x[8]})).collect::<Vec<_>>(),
            "issue_interval_histograms":self.intervals,"issue_stage_samples":self.samples,
            "stage_totals_count_operand_mac_rmw":self.phases,
            "hbm_sum":self.h.iter().sum::<u64>(),"core_sums":self.c.iter().map(|x|x.iter().sum::<u64>()).collect::<Vec<_>>(),
            "mutually_exclusive":self.h.iter().sum::<u64>()==cycles && self.c.iter().all(|x|x.iter().sum::<u64>()==cycles)})
    }
}
impl Sim {
    pub(super) fn profile_hbm_cycle(&mut self) {
        if !self.cfg.diagnostic_profile {
            return;
        }
        let p = &self.profile_v3;
        let fired =
            self.dma_serial > p.prev_dma || self.diagnostics.accepted_this_cycle.iter().any(|&x| x);
        let dma_ready = self.dma_ready_now();
        let blocked_slots = self
            .cores
            .iter()
            .any(|c| c.session.is_some() && c.free_slots.is_empty());
        let idx = if fired {
            0
        } else if !dma_ready {
            5
        } else if self.credit_used >= self.cfg.credits {
            if p.returned_pending > 0 { 2 } else { 1 }
        } else if blocked_slots {
            3
        } else {
            4
        };
        self.profile_v3.h[idx] += 1;
        self.profile_v3.prev_dma = self.dma_serial;
    }
    pub(super) fn profile_core_cycle(&mut self, c: usize, state: &str) {
        if !self.cfg.diagnostic_profile {
            return;
        }
        let idx = match state {
            "issue" => 1,
            "weight_not_ready" => {
                let sent = self.cores[c]
                    .run
                    .as_ref()
                    .and_then(|r| r.issues.get(r.pos).map(|m| r.tiles[m.tile].sent))
                    .unwrap_or(0);
                if sent > 0 { 2 } else { 3 }
            }
            "operand_feed" => 4,
            "x_not_ready" => 5,
            "previous_k_commit" | "result_context_full" | "chunk_result_drain" => 6,
            "chunk_conversion" | "vector" | "phase_advance" => 7,
            _ => {
                if self.cores[c].session.is_none() {
                    0
                } else if self.now < self.control_free {
                    8
                } else {
                    7
                }
            }
        };
        self.profile_v3.c[c][idx] += 1;
    }
    pub(super) fn profile_combine_tail(&mut self, start: u64, end: u64) {
        if !self.cfg.diagnostic_profile {
            return;
        }
        self.profile_v3.h[4] += end - start;
        for x in &mut self.profile_v3.c {
            x[7] += end - start
        }
    }
    pub(super) fn profile_weight_ready(
        &mut self,
        c: usize,
        task: usize,
        phase: usize,
        tile: usize,
    ) {
        if self.cfg.diagnostic_profile {
            self.profile_v3
                .ready
                .insert((c, task, phase * 1_000_000 + tile), self.now);
        }
    }
    pub(super) fn profile_return(&mut self) {
        if self.cfg.diagnostic_profile {
            self.profile_v3.returned_pending += 1
        }
    }
    pub(super) fn profile_landed(&mut self) {
        if self.cfg.diagnostic_profile {
            self.profile_v3.returned_pending -= 1
        }
    }
    pub(super) fn profile_issue(&mut self, c: usize, i: usize, tid: usize, xs: usize, end: u64) {
        if !self.cfg.diagnostic_profile {
            return;
        }
        let s = self.cores[c].session.as_ref().unwrap();
        let p = &mut self.profile_v3;
        if let Some(last) = p.prev_issue[c] {
            *p.intervals[c].entry(self.now - last).or_default() += 1;
        }
        p.prev_issue[c] = Some(self.now);
        p.pending.insert(
            (c, i),
            Timeline {
                core: c,
                task: s.e,
                phase: s.phase,
                issue: i,
                x_ready: self.cores[c].xs[xs].ready,
                w_ready: *p
                    .ready
                    .get(&(c, s.e, s.phase * 1_000_000 + tid))
                    .unwrap_or(&self.now),
                operand_start: self.now,
                operand_end: end,
                mac_start: end,
                mac_end: end + self.cfg.dot_tail_ns,
                ..Default::default()
            },
        );
    }
    pub(super) fn profile_dot(&mut self, c: usize, i: usize) {
        if self.cfg.diagnostic_profile {
            if let Some(p) = self.profile_v3.pending.get_mut(&(c, i)) {
                p.rmw_start = self.now
            }
        }
    }
    pub(super) fn profile_commit(&mut self, c: usize, i: usize) {
        if !self.cfg.diagnostic_profile {
            return;
        }
        if let Some(mut p) = self.profile_v3.pending.remove(&(c, i)) {
            p.accumulator_done = self.now;
            let x = self
                .profile_v3
                .phases
                .entry(format!("core{}:phase{}", c, p.phase))
                .or_default();
            x[0] += 1;
            x[1] += p.operand_end - p.operand_start;
            x[2] += p.mac_end - p.mac_start;
            x[3] += p.accumulator_done - p.rmw_start;
            if self.profile_v3.samples.len() < self.cfg.profile_issue_limit {
                self.profile_v3.samples.push(p)
            }
        }
    }
}
