//! Finite-window critical-anchor pairing. This is a charged, bounded heuristic,
//! not global scheduling or a predictor of which experts the router selects.
use super::*;

#[derive(Clone)]
struct Candidate {
    task: usize,
    potential: u8,
    eligible: u8, // Due now for irrevocable commitment.
    service: [u64; 2],
    finish: [u64; 2],
    age: u8,
}
#[derive(Clone, Copy)]
struct Placement {
    index: usize,
    core: usize,
}
pub(super) struct Snapshot {
    started: u64,
    ready: u64,
    candidates: Vec<Candidate>,
    proposed: Vec<Placement>,
    chosen: Vec<Placement>,
    anchor: usize,
    age_forced: bool,
    remaining: [u64; 2],
    comparisons: u64,
    charged_cycles: u64,
}
#[derive(Default, Serialize)]
pub(super) struct Diagnostics {
    pub snapshot_count: u64,
    pub snapshot_candidate_sum: u64,
    pub pair_comparisons: u64,
    pub control_cycles: u64,
    pub paired_rounds: u64,
    pub single_rounds: u64,
    pub reordered_bindings: u64,
    pub aging_forced_rounds: u64,
    pub max_bypass: u8,
    pub initial_fill_wait_cycles: u64,
    pub no_eligible_cycles: u64,
    pub revalidation_cancellations: u64,
    pub estimate_saturations: u64,
    pub late_floor_blocked_core_cycles: u64,
    pub deferred_placements: u64,
    pub preferred_core_wait_cycles: u64,
    pub audit: Vec<Value>, // Observer only: never read for hardware decisions.
}
#[derive(Default)]
pub(super) struct JointState {
    pub snapshot: Option<Snapshot>,
    // At most eight entries, physically stored in the existing 64B descriptors.
    pub ages: BTreeMap<usize, u8>,
    pub retry_after: u64, // Bounded backoff after an empty or invalid scan.
    pub deferred_core_mask: u8,
    pub deferred_generation: usize,
    pub diagnostics: Diagnostics,
}

fn propose(
    candidates: &[Candidate],
    nc: usize,
    age_limit: u8,
    pairing: bool,
    rr: usize,
) -> (Option<usize>, bool, Vec<Placement>, u64) {
    let min_service = |x: &Candidate| {
        (0..nc)
            .filter(|&c| x.potential & (1 << c) != 0)
            .map(|c| x.service[c])
            .min()
            .unwrap_or(0)
    };
    // Aging is admission-round fairness under continued legal opportunities,
    // not a guarantee in cycles while a task cannot fit or a core is busy.
    let aged = candidates
        .iter()
        .enumerate()
        .find(|(_, x)| x.potential != 0 && x.age >= age_limit)
        .map(|(i, _)| i);
    let anchor = aged.or_else(|| {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, x)| x.potential != 0)
            .max_by_key(|(i, x)| (min_service(x), Reverse(*i)))
            .map(|(i, _)| i)
    });
    let mut chosen = Vec::new();
    let mut comparisons = 0;
    if let Some(a) = anchor {
        let x = &candidates[a];
        // Maximum cardinality first: a singleton and a pair do not perform
        // equal work. Compare pairs only when any feasible distinct pair exists.
        let mut best_pair = None;
        if pairing && nc == 2 {
            for c in 0..2 {
                if x.potential & (1 << c) == 0 {
                    continue;
                }
                let other = 1 - c;
                for (j, y) in candidates.iter().enumerate() {
                    if j == a || y.potential & (1 << other) == 0 {
                        continue;
                    }
                    comparisons += 1;
                    let score = (
                        x.finish[c].max(y.finish[other]),
                        Reverse(min_service(x) + min_service(y)),
                        x.finish[c] + y.finish[other],
                        c,
                        j,
                    );
                    if best_pair.as_ref().is_none_or(|(old, _)| score < *old) {
                        best_pair = Some((
                            score,
                            vec![
                                Placement { index: a, core: c },
                                Placement {
                                    index: j,
                                    core: other,
                                },
                            ],
                        ));
                    }
                }
            }
        }
        if let Some((_, pair)) = best_pair {
            chosen = pair;
        } else {
            let c = (0..nc)
                .filter(|&c| x.potential & (1 << c) != 0)
                .min_by_key(|&c| (x.finish[c], (c + nc - rr) % nc))
                .unwrap();
            comparisons += x.potential.count_ones() as u64;
            chosen.push(Placement { index: a, core: c });
        }
    }
    (anchor, aged.is_some(), chosen, comparisons)
}

// A bounded two-ended policy: a high-intensity descriptor targets the wider
// compute core while the lowest-intensity visible companion targets the other.
// Both ends are selected only from the same already-arrived eight descriptors.
// Existing capacity masks and late admission still govern irrevocable binding.
fn propose_ipd(
    candidates: &[Candidate],
    experts: &[Expert],
    core_m: &[usize],
    age_limit: u8,
    rr: usize,
) -> (Option<usize>, bool, Vec<Placement>, u64) {
    let nc = core_m.len();
    let aged = candidates
        .iter()
        .enumerate()
        .find(|(_, x)| x.potential != 0 && x.age >= age_limit)
        .map(|(i, _)| i);
    let anchor = aged.or_else(|| {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, x)| x.potential != 0)
            .max_by_key(|(i, x)| {
                let e = &experts[x.task];
                (e.is_shared as u8, e.m, Reverse(*i))
            })
            .map(|(i, _)| i)
    });
    let mut comparisons = 0;
    let Some(a) = anchor else {
        return (None, false, vec![], comparisons);
    };
    let x = &candidates[a];
    let wide = (0..nc).max_by_key(|&c| (core_m[c], Reverse(c))).unwrap();
    let preferred = if x.potential & (1 << wide) != 0 {
        wide
    } else {
        (0..nc)
            .filter(|&c| x.potential & (1 << c) != 0)
            .min_by_key(|&c| (x.finish[c], (c + nc - rr) % nc))
            .unwrap()
    };
    if nc == 1 {
        return (
            anchor,
            aged.is_some(),
            vec![Placement {
                index: a,
                core: preferred,
            }],
            1,
        );
    }
    // When shapes are equal, both orientations are legal; keep the better
    // finish-time orientation instead of inventing a "wide" homogeneous core.
    let orientations: Vec<usize> = if core_m[0] == core_m[1] {
        (0..nc).collect()
    } else {
        vec![preferred]
    };
    let mut best: Option<((u64, usize, usize, usize), Vec<Placement>)> = None;
    for c in orientations {
        if x.potential & (1 << c) == 0 {
            continue;
        }
        let other = 1 - c;
        let low = candidates
            .iter()
            .enumerate()
            .filter(|(j, y)| *j != a && y.potential & (1 << other) != 0)
            .min_by_key(|(j, y)| (experts[y.task].m, y.finish[other], *j));
        if let Some((j, y)) = low {
            comparisons += candidates.len() as u64;
            let score = (x.finish[c].max(y.finish[other]), experts[y.task].m, c, j);
            let pair = vec![
                Placement { index: a, core: c },
                Placement {
                    index: j,
                    core: other,
                },
            ];
            if best.as_ref().is_none_or(|(old, _)| score < *old) {
                best = Some((score, pair));
            }
        }
    }
    let proposed = if let Some((_, pair)) = best {
        pair
    } else {
        comparisons += nc as u64;
        vec![Placement {
            index: a,
            core: preferred,
        }]
    };
    (anchor, aged.is_some(), proposed, comparisons)
}

impl Sim {
    fn joint_lead_bound(&self, visible: usize) -> u64 {
        // Read/shape scan + at most two orientations per visible descriptor +
        // two commits. Paid service may be smaller; admission uses this bound.
        if !self.cfg.control_cost {
            return 0;
        }
        4 + 4 * visible as u64 * self.cores.len() as u64 + 4 * visible as u64 + 8
    }
    fn joint_legal(&self, e: usize, c: usize, lead: u64) -> bool {
        let core = &self.cores[c];
        core.next.is_none()
            && self.fits(e, c, false)
            && !core.free_slots.is_empty()
            && (!self.cfg.joint_late_bind
                || core.session.is_none()
                || self.progress_estimate(c)
                    <= self
                        .first_ready_estimate(e, c)
                        .saturating_add(lead)
                        .saturating_add(self.cfg.joint_margin_cycles))
    }
    fn joint_scan_opportunity(&self, lead: u64) -> bool {
        // Cheap per-core gate using the maximum first tile (4KiB), not a free
        // per-cycle descriptor search. Exact per-task eligibility is scanned.
        let bw = self
            .cfg
            .hbm_bytes_per_ns
            .min(self.cfg.credits * 32 / self.cfg.hbm_latency_ns.max(1) as usize)
            .max(1);
        self.cores.iter().enumerate().any(|(c, core)| {
            if core.next.is_some()
                || core.free_slots.is_empty()
                || (self.joint.deferred_core_mask != 0
                    && self.joint.deferred_core_mask & (1 << c) == 0)
            {
                return false;
            }
            let fetch = if self.cfg.ideal_hbm {
                0
            } else {
                self.cfg.hbm_latency_ns + ceil(4096 * self.cores.len(), bw) as u64
            };
            let landing = if self.cfg.ideal_onchip {
                0
            } else {
                ceil(4096, core.wb.free.len() * 16) as u64
            };
            !self.cfg.joint_late_bind
                || core.session.is_none()
                || self.progress_estimate(c)
                    <= fetch + landing + lead + self.cfg.joint_margin_cycles
        })
    }
    fn joint_saturate(&mut self, value: u64) -> u64 {
        if value > u32::MAX as u64 {
            self.joint.diagnostics.estimate_saturations += 1;
        }
        value.min(u32::MAX as u64)
    }
    fn joint_standalone(&self, e: usize, c: usize) -> u64 {
        let value = self.prediction(e, c, 1, false);
        if self.feedback_enabled() {
            value
                .saturating_mul(self.feedback_q8[c][Self::feedback_bin(self.w.experts[e].m)])
                .div_ceil(256)
        } else {
            value
        }
    }
    pub(super) fn dispatch_joint(&mut self) {
        if let Some(snapshot) = &self.joint.snapshot {
            if self.now < snapshot.ready {
                return;
            }
            let snapshot = self.joint.snapshot.take().unwrap();
            self.commit_joint(snapshot);
            return;
        }
        let initial = self.cfg.window.min(self.w.experts.len());
        if self.input_cursor < initial {
            self.joint.diagnostics.initial_fill_wait_cycles += 1;
            return;
        }
        if self.now < self.control_free || self.now < self.joint.retry_after {
            return;
        }
        if self.joint.deferred_generation != self.input_cursor {
            self.joint.deferred_core_mask = 0;
        }
        let visible = self.pending.len();
        if visible == 0 {
            return;
        }
        let lead = self.joint_lead_bound(visible);
        if self.cfg.joint_late_bind {
            for (c, core) in self.cores.iter().enumerate() {
                if let Some(session) = &core.session {
                    let e = &self.w.experts[session.e];
                    let floor = ceil(4 * e.m * e.h, self.cfg.onchip_bytes_per_ns) as u64
                        + self.cfg.dot_tail_ns
                        + core.contexts as u64 * 5;
                    if core.next.is_none()
                        && floor
                            > self.first_ready_estimate(session.e, c)
                                + lead
                                + self.cfg.joint_margin_cycles
                    {
                        self.joint.diagnostics.late_floor_blocked_core_cycles += 1;
                    }
                }
            }
        }
        if !self.joint_scan_opportunity(lead) {
            self.joint.diagnostics.no_eligible_cycles += 1;
            self.joint.diagnostics.preferred_core_wait_cycles +=
                u64::from(self.joint.deferred_core_mask != 0);
            return;
        }
        let tasks: Vec<_> = self.pending.iter().copied().collect();
        assert!(tasks.len() <= 8 && self.joint.ages.len() <= 8);
        let mut candidates = Vec::with_capacity(tasks.len());
        for e in tasks {
            let mut candidate = Candidate {
                task: e,
                potential: 0,
                eligible: 0,
                service: [0; 2],
                finish: [0; 2],
                age: *self.joint.ages.get(&e).unwrap(),
            };
            for c in 0..self.cores.len() {
                if self.cores[c].next.is_none() && self.fits(e, c, false) {
                    candidate.potential |= 1 << c;
                }
                if self.joint_legal(e, c, lead) {
                    candidate.eligible |= 1 << c;
                }
                candidate.service[c] = self.joint_saturate(self.joint_standalone(e, c));
                candidate.finish[c] = self.joint_saturate(self.predicted_finish_delay(e, c));
            }
            candidates.push(candidate);
        }
        let (anchor, age_forced, proposed, comparisons) = if self.cfg.dispatch == "ipd" {
            let core_m: Vec<_> = self.cores.iter().map(|c| c.m).collect();
            propose_ipd(
                &candidates,
                &self.w.experts,
                &core_m,
                self.cfg.joint_age_limit,
                self.rr_desc,
            )
        } else {
            propose(
                &candidates,
                self.cores.len(),
                self.cfg.joint_age_limit,
                self.cfg.joint_pairing,
                self.rr_desc,
            )
        };
        // Compare complete future pairs, then bind only placements whose DMA
        // lead window has opened. A hot task is not forced onto an idle small
        // core merely because the preferred large core is not yet due.
        let chosen: Vec<_> = proposed
            .iter()
            .copied()
            .filter(|p| candidates[p.index].eligible & (1 << p.core) != 0)
            .collect();
        let cost = if self.cfg.control_cost {
            4 + 4 * visible as u64 * self.cores.len() as u64
                + 2 * comparisons
                + 4 * chosen.len() as u64
        } else {
            0
        };
        assert!(cost <= lead);
        self.control_free = self.now + cost;
        self.cores[0].stats.control_cycles += cost;
        self.joint.diagnostics.control_cycles += cost;
        self.joint.diagnostics.snapshot_count += 1;
        self.joint.diagnostics.snapshot_candidate_sum += visible as u64;
        self.joint.diagnostics.pair_comparisons += comparisons;
        let mut remaining = [0; 2];
        for (c, r) in remaining.iter_mut().enumerate().take(self.cores.len()) {
            *r = self.progress_estimate(c).min(u32::MAX as u64);
        }
        let snapshot = Snapshot {
            started: self.now,
            ready: self.now + cost,
            candidates,
            proposed,
            chosen,
            anchor: anchor.unwrap_or(usize::MAX),
            age_forced,
            remaining,
            comparisons,
            charged_cycles: cost,
        };
        // Even a scan finding no legal candidate pays its complete read service.
        self.joint.snapshot = Some(snapshot);
    }
    fn commit_joint(&mut self, snapshot: Snapshot) {
        let candidates = &snapshot.candidates;
        let chosen = &snapshot.chosen;
        // Revalidate hard resources, not the predictive late threshold. Elapsed
        // control service need not advance issue-count progress; cancelling on
        // that estimate would repeatedly monopolize the control port.
        let valid = !chosen.is_empty()
            && chosen.iter().all(|p| {
                let e = candidates[p.index].task;
                self.status[e] == 0
                    && self.pending.contains(&e)
                    && self.cores[p.core].next.is_none()
                    && self.fits(e, p.core, false)
                    && !self.cores[p.core].free_slots.is_empty()
            });
        let audit = json!({"policy":self.cfg.dispatch,"snapshot_cycle":snapshot.started,"commit_cycle":self.now,
            "charged_cycles":snapshot.charged_cycles,"comparisons":snapshot.comparisons,
            "anchor":candidates.get(snapshot.anchor).map(|x|x.task),"age_forced":snapshot.age_forced,
            "candidates":candidates.iter().map(|x|json!({"task":x.task,"age":x.age,
                "potential_mask":x.potential,"eligible_mask":x.eligible,"standalone_service":&x.service[..self.cores.len()],
                "finish_delay":&x.finish[..self.cores.len()]})).collect::<Vec<_>>(),
            "proposed":snapshot.proposed.iter().map(|p|json!({"task":candidates[p.index].task,"core":p.core,"due":candidates[p.index].eligible&(1<<p.core)!=0})).collect::<Vec<_>>(),
            "chosen":chosen.iter().map(|p|json!({"task":candidates[p.index].task,"core":p.core})).collect::<Vec<_>>(),
            "committed":valid});
        self.joint.diagnostics.audit.push(audit);
        self.joint.diagnostics.deferred_placements +=
            (snapshot.proposed.len() - chosen.len()) as u64;
        self.joint.deferred_generation = self.input_cursor;
        self.joint.deferred_core_mask = if chosen.is_empty() {
            snapshot
                .proposed
                .iter()
                .fold(0, |mask, p| mask | (1 << p.core))
        } else {
            0
        };
        if !valid {
            if !chosen.is_empty() {
                self.joint.diagnostics.revalidation_cancellations += 1;
            } else {
                self.joint.diagnostics.no_eligible_cycles += self.now - snapshot.started + 1;
            }
            self.joint.retry_after = self.now + snapshot.charged_cycles.max(1);
            return;
        }
        assert!(chosen.len() <= 2);
        if chosen.len() == 2 {
            assert_ne!(chosen[0].core, chosen[1].core);
            assert_ne!(
                candidates[chosen[0].index].task,
                candidates[chosen[1].index].task
            );
            self.joint.diagnostics.paired_rounds += 1;
        } else {
            self.joint.diagnostics.single_rounds += 1;
        }
        self.joint.diagnostics.aging_forced_rounds += u64::from(snapshot.age_forced);
        let selected_tasks: Vec<_> = chosen.iter().map(|p| candidates[p.index].task).collect();
        for p in chosen {
            let x = &candidates[p.index];
            let e = x.task;
            let c = p.core;
            let index = self.pending.iter().position(|&task| task == e).unwrap();
            self.joint.diagnostics.reordered_bindings += u64::from(index != 0);
            assert_eq!(self.pending.remove(index), Some(e));
            self.joint.ages.remove(&e);
            self.status[e] = 3;
            self.cores[c].next = Some(NextTask { e, first: None });
            // Commit service includes two owner cycles and two slot-install
            // cycles. Reserve atomically before any Current admission can run.
            self.reserve_next_tile(c, self.now);
            self.cores[c].stats.next_bindings += 1;
            self.decisions += 1;
            self.last_progress = self.now;
            self.rr_desc = (c + 1) % self.cores.len();
            let eligible: Vec<_> = (0..self.cores.len())
                .filter(|&n| x.eligible & (1 << n) != 0)
                .collect();
            self.dispatch_audit.push(json!({"cycle":self.now,"task":e,"expert":self.w.experts[e].id,
                "core":c,"eligible_cores":eligible,"joint":true,"snapshot_cycle":snapshot.started,
                "snapshot_candidates":candidates.len(),"anchor":candidates[snapshot.anchor].task,
                "chosen_pair":selected_tasks,"predicted_service_snapshot":x.service[c],
                "predicted_finish_cycle":snapshot.ready+x.finish[c],
                "remaining_estimate":snapshot.remaining[c],"first_ready_delay_estimate":self.first_ready_estimate(e,c),
                "service_estimate":self.prediction(e,c,self.cores.len(),false)}));
            trace!(
                self,
                json!({"event":"commit_owner","cycle":self.now,"task":e,
                "expert":self.w.experts[e].id,"core":c,"split":false,"joint":true,
                "remaining_estimate":snapshot.remaining[c],"first_ready_delay_estimate":self.first_ready_estimate(e,c),
                "service_estimate":self.prediction(e,c,self.cores.len(),false)})
            );
        }
        for age in self.joint.ages.values_mut() {
            *age = age.saturating_add(1);
            self.joint.diagnostics.max_bypass = self.joint.diagnostics.max_bypass.max(*age);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sim(ms: &[usize]) -> Sim {
        // Runtime fixture; production construction separately validates Compiler
        // reservations. Start from legacy physical parameters then select policy.
        let batch = *ms.iter().max().unwrap();
        let workload = Workload {
            id: "joint-protocol".into(),
            batch,
            hidden: 512,
            top_k: ms.len(),
            engine_layout: Value::Null,
            experts: ms
                .iter()
                .enumerate()
                .map(|(e, &m)| Expert {
                    id: e as i64,
                    is_shared: false,
                    m,
                    h: 512,
                    f: 128,
                    token_indices: (0..m).collect(),
                    weights: Value::Null,
                })
                .collect(),
        };
        let mut s = Sim::new(
            workload,
            Config {
                record_trace: true,
                ..Default::default()
            },
        );
        s.cfg.dispatch = "joint".into();
        s.cfg.joint_feedback = false;
        s
    }
    fn prepare_snapshot(s: &mut Sim) {
        let initial = s.cfg.window.min(s.w.experts.len());
        for t in 0..initial {
            s.now = t as u64;
            s.dispatch_runtime();
        }
        assert!(s.joint.snapshot.is_some());
    }
    #[test]
    fn initial_fill_paid_scan_and_atomic_slot_reservation() {
        let mut s = sim(&[1, 2, 32]);
        s.dispatch_runtime();
        assert!(s.joint.snapshot.is_none());
        s.now = 1;
        s.dispatch_runtime();
        assert!(s.joint.snapshot.is_none());
        s.now = 2;
        s.dispatch_runtime();
        let snapshot = s.joint.snapshot.as_ref().unwrap();
        assert_eq!(snapshot.candidates.len(), 3);
        assert_eq!(snapshot.chosen.len(), 2);
        assert_eq!(snapshot.candidates[snapshot.anchor].task, 2);
        let due = snapshot.ready;
        assert!(due > s.now);
        s.now = due - 1;
        s.dispatch_runtime();
        assert!(s.cores.iter().all(|c| c.next.is_none()));
        s.now = due;
        s.dispatch_runtime();
        assert_eq!(s.joint.diagnostics.paired_rounds, 1);
        let owners: Vec<_> = s.cores.iter().map(|c| c.next.as_ref().unwrap().e).collect();
        assert_ne!(owners[0], owners[1]);
        assert_eq!(s.pending.len(), 1);
        assert!(s.pending.iter().all(|e| s.joint.ages[e] == 1));
        for core in &s.cores {
            assert_eq!(core.free_slots.len(), core.wslots - 1);
            assert!(core.next.as_ref().unwrap().first.is_some());
            assert_eq!(core.stats.weight_bytes, 4096);
        }
        assert_eq!(s.credit_used, 0);
    }
    #[test]
    fn wait_for_preferred_core_does_not_force_hot_task_to_idle_small() {
        let entries = vec![
            Candidate {
                task: 0,
                potential: 3,
                eligible: 2,
                service: [1000, 2000],
                finish: [1200, 2000],
                age: 0,
            },
            Candidate {
                task: 1,
                potential: 3,
                eligible: 2,
                service: [100, 100],
                finish: [300, 100],
                age: 0,
            },
        ];
        let (anchor, _, proposal, _) = propose(&entries, 2, 8, true, 0);
        assert_eq!(anchor, Some(0));
        assert_eq!(proposal.len(), 2);
        assert_eq!((proposal[0].index, proposal[0].core), (0, 0));
        assert_eq!((proposal[1].index, proposal[1].core), (1, 1));
        let due: Vec<_> = proposal
            .iter()
            .filter(|p| entries[p.index].eligible & (1 << p.core) != 0)
            .collect();
        assert_eq!(due.len(), 1);
        assert_eq!((due[0].index, due[0].core), (1, 1));
    }
    #[test]
    fn aged_descriptor_anchors_charged_scan() {
        let mut s = sim(&[1, 2, 32]);
        s.input_cursor = 3;
        s.pending = (0..3).collect();
        s.joint.ages = [(0, 8), (1, 0), (2, 0)].into_iter().collect();
        s.dispatch_joint();
        let shot = s.joint.snapshot.as_ref().unwrap();
        assert_eq!(shot.candidates[shot.anchor].task, 0);
        assert!(shot.age_forced);
        assert!(shot.charged_cycles >= 4 + 4 * 3 * 2 + 4 * 2);
    }
    #[test]
    fn failed_resource_revalidation_cannot_bind_or_start_dma() {
        let mut s = sim(&[1, 32]);
        prepare_snapshot(&mut s);
        let due = s.joint.snapshot.as_ref().unwrap().ready;
        s.cores[0].free_slots.clear();
        s.now = due;
        s.dispatch_joint();
        assert_eq!(s.joint.diagnostics.revalidation_cancellations, 1);
        assert!(s.cores.iter().all(|c| c.next.is_none()));
        assert_eq!(s.status, vec![0, 0]);
        assert_eq!(s.credit_used, 0);
        assert!(s.joint.retry_after > s.now);
    }
    #[test]
    fn joint_whole_ffn_drains_repeats_and_preserves_bytes() {
        let a = sim(&[1, 2, 8, 3]).run();
        let b = sim(&[1, 2, 8, 3]).run();
        assert_eq!(a, b);
        assert_eq!(a["drained"], true);
        assert_eq!(a["joint_diagnostics"]["estimate_saturations"], 0);
        assert_eq!(a["dispatch_audit"].as_array().unwrap().len(), 4);
        assert_eq!(a["dma_transactions_accepted"], a["dma_transactions_landed"]);
        let mut baseline = sim(&[1, 2, 8, 3]);
        baseline.cfg.dispatch = "dynamic".into();
        let old = baseline.run();
        assert_eq!(a["weight_bytes"], old["weight_bytes"]);
        assert_eq!(a["useful_macs"], old["useful_macs"]);
    }
    #[test]
    fn pairing_and_feedback_are_independent_switches() {
        let mut s = sim(&[1, 2, 8, 3]);
        s.cfg.joint_pairing = false;
        s.cfg.joint_feedback = true;
        let report = s.run();
        assert_eq!(report["joint_diagnostics"]["paired_rounds"], 0);
        assert_eq!(report["joint_diagnostics"]["single_rounds"], 4);
        assert_eq!(report["feedback_updates"], 4);
    }
    #[test]
    fn no_prefetch_reserves_slot_but_requests_wait_for_promotion() {
        let mut s = sim(&[1, 2]);
        s.cfg.next_prefetch = false;
        prepare_snapshot(&mut s);
        s.now = s.joint.snapshot.as_ref().unwrap().ready;
        s.dispatch_runtime();
        assert!(
            s.cores
                .iter()
                .all(|c| c.next.as_ref().unwrap().first.is_some())
        );
        s.issue_runtime();
        assert_eq!(s.credit_used, 0);
        let report = sim(&[1, 2]).run();
        let mut no = sim(&[1, 2]);
        no.cfg.next_prefetch = false;
        let no = no.run();
        assert_eq!(report["weight_bytes"], no["weight_bytes"]);
        assert_eq!(no["drained"], true);
    }
    #[test]
    fn two_ended_policy_places_hot_shared_and_cold_companion() {
        let mut s = sim(&[1, 2, 16, 3]);
        s.w.experts[2].is_shared = true;
        let candidates: Vec<_> = (0..4)
            .map(|task| Candidate {
                task,
                potential: 3,
                eligible: 3,
                service: [100 + task as u64; 2],
                finish: [100 + task as u64; 2],
                age: 0,
            })
            .collect();
        let (anchor, aged, placements, _) = propose_ipd(&candidates, &s.w.experts, &[4, 2], 8, 0);
        assert_eq!(anchor, Some(2));
        assert!(!aged);
        assert_eq!(
            placements
                .iter()
                .map(|p| (p.index, p.core))
                .collect::<Vec<_>>(),
            vec![(2, 0), (0, 1)]
        );
    }
    #[test]
    fn ipd_keeps_work_bytes_and_finite_credits() {
        let mut s = sim(&[1, 2, 8, 3]);
        s.cfg.dispatch = "ipd".into();
        s.cfg.ipd_credit_quotas = true;
        let result = s.run();
        assert_eq!(result["drained"], true);
        assert_eq!(result["credit_peak"].as_u64().unwrap() <= 256, true);
        assert_eq!(
            result["dma_transactions_accepted"],
            result["dma_transactions_landed"]
        );
        let baseline = sim(&[1, 2, 8, 3]).run();
        assert_eq!(result["weight_bytes"], baseline["weight_bytes"]);
        assert_eq!(result["useful_macs"], baseline["useful_macs"]);
        assert_eq!(
            result["joint_diagnostics"]["paired_rounds"]
                .as_u64()
                .unwrap()
                > 0,
            true
        );
    }
}
