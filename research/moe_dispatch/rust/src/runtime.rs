//! Bounded whole-expert Current/Next controller. All completion and readiness
//! decisions use real events; estimates are used only to choose an owner.
use super::*;

pub(super) struct NextTask {
    pub e: usize,
    pub first: Option<Tile>,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sim(credits: usize) -> Sim {
        let w = Workload {
            id: "protocol".into(),
            batch: 1,
            hidden: 512,
            top_k: 1,
            engine_layout: Value::Null,
            experts: vec![Expert {
                id: 0,
                is_shared: false,
                m: 1,
                h: 512,
                f: 16,
                token_indices: vec![0],
                weights: Value::Null,
            }],
        };
        let mut s = Sim::new(
            w,
            Config {
                credits,
                control_cost: false,
                ..Default::default()
            },
        );
        s.status[0] = 3;
        s.cores[0].next = Some(NextTask { e: 0, first: None });
        s.admit_next_weights();
        s
    }
    #[test]
    fn valid_under_backpressure_is_stable_and_not_charged() {
        let mut s = sim(2);
        s.cfg.dma_ready_after = 20;
        s.issue_runtime();
        let request = s.pending_dma.clone().unwrap();
        for t in 1..20 {
            s.now = t;
            s.issue_runtime();
            assert_eq!(s.pending_dma, Some(request.clone()));
        }
        assert_eq!(s.credit_used, 0);
        assert_eq!(
            s.cores[0]
                .next
                .as_ref()
                .unwrap()
                .first
                .as_ref()
                .unwrap()
                .sent,
            0
        );
        s.now = 20;
        s.issue_runtime();
        assert_eq!(s.credit_used, 2);
        assert_eq!(s.outstanding_dma.get(&request.serial), Some(&request));
    }
    #[test]
    fn credit_held_until_landing_then_same_cycle_reusable() {
        let mut s = sim(1);
        s.issue_runtime();
        let request = s.outstanding_dma.values().next().unwrap().clone();
        s.issue_runtime();
        assert_eq!(s.dma_serial, 1);
        // A response reaches the SRAM interface, but its bank write has not completed.
        s.now = 64;
        s.return_runtime(request.clone());
        assert_eq!(s.credit_used, 1);
        s.now = 65;
        s.ack_runtime(request);
        assert_eq!(s.credit_used, 0);
        s.issue_runtime();
        assert_eq!(s.credit_used, 1);
        assert_eq!(s.dma_serial, 2);
    }
    #[test]
    fn in_flight_next_response_survives_promotion_and_input_gather() {
        let mut s = sim(8);
        s.issue_runtime();
        let request = s.outstanding_dma.values().next().unwrap().clone();
        assert!(s.cores[0].session.is_none());
        s.promote_runtime();
        assert!(s.cores[0].next.is_none());
        assert!(s.cores[0].incoming.is_some());
        s.ack_runtime(request);
        assert_eq!(s.cores[0].incoming.as_ref().unwrap().acks, 32);
        assert_eq!(s.cores[0].stats.next_inflight_at_promotion, 1);
    }
    #[test]
    fn urgent_without_legal_request_has_no_grant() {
        let mut s = sim(8);
        let t = s.cores[0].next.as_mut().unwrap().first.as_mut().unwrap();
        t.sent = t.bytes;
        t.acks = t.bytes;
        t.ready = true;
        s.cores[0].dma_wait_since = Some(0);
        s.now = 1000;
        s.issue_runtime();
        assert_eq!(s.dma_serial, 0);
        assert!(s.pending_dma.is_none());
    }
    #[test]
    fn duplicate_return_is_rejected() {
        let mut s = sim(8);
        s.issue_runtime();
        let r = s.outstanding_dma.values().next().unwrap().clone();
        s.ack_runtime(r.clone());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.ack_runtime(r))).is_err()
        );
    }
    #[test]
    fn next_reserves_one_tile_only_even_with_many_free_slots() {
        let mut s = sim(8);
        for t in 0..100 {
            s.now = t;
            s.admit_next_weights();
        }
        assert_eq!(s.cores[0].stats.next_prefetch_tiles, 1);
        assert_eq!(s.cores[0].free_slots.len(), s.cores[0].wslots - 1);
    }
    #[test]
    #[should_panic(expected = "no progress")]
    fn stalled_machine_reports_wait_graph_instead_of_hanging() {
        let s = sim(8);
        let cfg = Config {
            no_progress_cycles: 50,
            dma_ready_after: 10000,
            ..Default::default()
        };
        Sim::new(s.w, cfg).run();
    }
    #[test]
    fn completed_promotion_is_not_delayed_by_a_later_control_service() {
        let mut s = sim(8);
        s.cfg.control_cost = true;
        s.promote_runtime();
        assert_eq!(s.promotion_pending[0], Some(2));
        s.now = 2;
        s.control_free = 100; // An unrelated later lease must not retime completion.
        s.finish_runtime_promotions();
        assert!(s.cores[0].next.is_none());
        assert!(s.cores[0].session.is_some());
        assert_eq!(s.promotion_pending[0], None);
    }
}

/// Stable identity survives Next -> Current and Current input gathering.
/// A task/phase/tile is admitted only once in this finite layer execution.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(super) struct DmaRequest {
    pub serial: u64,
    pub core: usize,
    pub task: usize,
    pub phase: usize,
    pub tile: usize,
    pub slot: usize,
    pub offset: usize,
    pub address: usize,
}

impl Sim {
    pub(super) fn wait_snapshot(&self) -> Value {
        json!({"cycle":self.now,"credit_used":self.credit_used,"pending_dma":self.pending_dma,
            "input_cursor":self.input_cursor,"pending":self.pending,
            "cores":self.cores.iter().map(|c|json!({
                "current":c.session.as_ref().map(|s|(s.e,s.phase)),
                "next":c.next.as_ref().map(|n|n.e),"free_w_slots":c.free_slots.len(),
                "result_contexts":c.contexts,"run":c.run.as_ref().map(|r|(r.pos,r.issues.len(),r.admit,r.send_cursor))
            })).collect::<Vec<_>>()})
    }

    fn issue_count(&self, e: usize, c: usize) -> u64 {
        let x = &self.w.experts[e];
        (ceil(x.m, self.cores[c].m)
            * (2 * ceil(x.f, 4) * ceil(x.h, 512) + ceil(x.h, 4) * ceil(x.f, 512))) as u64
    }

    fn progress_estimate(&self, c: usize) -> u64 {
        let Some(s) = &self.cores[c].session else {
            return 0;
        };
        let e = &self.w.experts[s.e];
        let total = self.issue_count(s.e, c).max(1);
        let issued = self.cores[c].stats.issues - s.issues_at_start;
        let remaining = total.saturating_sub(issued);
        // Progress counter, not original predicted duration minus wall time.
        // Future events/bank completion timestamps are deliberately not consulted.
        let service = self.prediction(s.e, c, self.cores.len(), false);
        let drain = ceil(4 * e.m * e.h, self.cfg.onchip_bytes_per_ns) as u64
            + self.cfg.dot_tail_ns
            + self.cores[c].contexts as u64 * 5;
        (service.saturating_mul(remaining).div_ceil(total)).max(drain)
    }

    fn first_ready_estimate(&self, e: usize, c: usize) -> u64 {
        let x = &self.w.experts[e];
        let bytes = x.f.min(4) * align(x.h.min(512) * 2, 32);
        let bw = self
            .cfg
            .hbm_bytes_per_ns
            .min(self.cfg.credits * 32 / (self.cfg.hbm_latency_ns.max(1) as usize))
            .max(1);
        let fetch = if self.cfg.ideal_hbm {
            0
        } else {
            self.cfg.hbm_latency_ns + ceil(bytes * self.cores.len(), bw) as u64
        };
        let landing = if self.cfg.ideal_onchip {
            0
        } else {
            ceil(bytes, self.cores[c].wb.free.len() * 16) as u64
        };
        fetch + landing
    }

    pub(super) fn dispatch_runtime(&mut self) {
        assert_eq!(
            self.cfg.split, "none",
            "Current/Next supports whole experts only; legacy split requires runtime_fsm=false"
        );
        if self.input_cursor < self.w.experts.len() && self.pending.len() < self.cfg.window {
            // One descriptor arrives per cycle from the finite upstream route
            // table already charged in the compiler arena. No future routes read.
            self.pending.push_back(self.input_cursor);
            self.input_cursor += 1;
            self.last_progress = self.now;
        } else if self.input_cursor < self.w.experts.len() {
            self.input_backpressure_cycles += 1;
        }
        self.unassigned_peak = self.unassigned_peak.max(self.pending.len());
        let Some(&e) = self.pending.front() else {
            return;
        };
        let eligible: Vec<_> = (0..self.cores.len())
            .filter(|&c| {
                self.cores[c].next.is_none()
                    && self.fits(e, c, false)
                    && (self.cfg.dispatch != "fixed" || self.fixed[e] == c)
            })
            .collect();
        if eligible.is_empty() || self.now < self.control_free {
            return;
        }
        if !self.runtime_decision_paid && self.cfg.control_cost {
            // 2 descriptor cycles, 4 per-core service/shape reads, comparison,
            // then owner update; no free full-window or full-expert search.
            let cost = 4 + 4 * eligible.len() as u64;
            self.control_free = self.now + cost;
            self.cores[eligible[0]].stats.control_cycles += cost;
            self.runtime_decision_paid = true;
            return;
        }
        self.runtime_decision_paid = false;
        let owner = *eligible
            .iter()
            .min_by_key(|&&c| {
                if matches!(self.cfg.dispatch.as_str(), "fifo" | "fixed") {
                    // Same task stream, work-conserving baseline; no shape-based affinity.
                    (
                        self.cores[c].session.is_some() as u64,
                        (c + self.cores.len() - self.rr_desc) % self.cores.len(),
                    )
                } else {
                    let available = self.progress_estimate(c);
                    let supply = self.first_ready_estimate(e, c);
                    let start = if self.cfg.next_prefetch {
                        available.max(supply)
                    } else {
                        available + supply
                    };
                    // Suffix includes ongoing memory throughput, operand banks,
                    // vector work and dependency tails: not compute-only after first tile.
                    let suffix = self
                        .prediction(e, c, self.cores.len(), false)
                        .saturating_sub(supply);
                    (
                        start + suffix,
                        (c + self.cores.len() - self.rr_desc) % self.cores.len(),
                    )
                }
            })
            .unwrap();
        assert_eq!(self.status[e], 0);
        assert_eq!(self.pending.pop_front(), Some(e));
        self.status[e] = 3; // Bound Next, not executing. Exactly one context owns e.
        self.cores[owner].next = Some(NextTask { e, first: None });
        self.cores[owner].stats.next_bindings += 1;
        self.decisions += 1;
        self.last_progress = self.now;
        self.rr_desc = (owner + 1) % self.cores.len();
        self.dispatch_audit.push(
            json!({"cycle":self.now,"task":e,"expert":self.w.experts[e].id,
            "core":owner,"eligible_cores":eligible,
            "remaining_estimate":self.progress_estimate(owner),
            "first_ready_delay_estimate":self.first_ready_estimate(e,owner),
            "service_estimate":self.prediction(e,owner,self.cores.len(),false)}),
        );
        self.log(json!({"event":"commit_owner","cycle":self.now,"task":e,
            "expert":self.w.experts[e].id,"core":owner,"split":false,
            "remaining_estimate":self.progress_estimate(owner),
            "first_ready_delay_estimate":self.first_ready_estimate(e,owner),
            "service_estimate":self.prediction(e,owner,self.cores.len(),false)}));
    }

    pub(super) fn promote_runtime(&mut self) {
        for c in 0..self.cores.len() {
            if self.cores[c].session.is_some()
                || self.cores[c].next.is_none()
                || self.promotion_pending[c].is_some()
            {
                continue;
            }
            if self.cfg.control_cost {
                if self.now < self.control_free {
                    continue;
                }
                self.control_free = self.now + 2;
                self.cores[c].stats.control_cycles += 2;
                self.promotion_pending[c] = Some(self.now + 2);
            } else {
                self.promotion_pending[c] = Some(self.now);
            }
        }
        self.finish_runtime_promotions();
    }

    pub(super) fn finish_runtime_promotions(&mut self) {
        for c in 0..self.cores.len() {
            if !self.promotion_pending[c].is_some_and(|done| done <= self.now) {
                continue;
            }
            self.promotion_pending[c] = None;
            assert!(self.cores[c].session.is_none());
            let next = self.cores[c].next.take().unwrap();
            assert_eq!(self.status[next.e], 3);
            self.status[next.e] = 1;
            if let Some(tile) = &next.first {
                self.cores[c].stats.next_ready_at_promotion += usize::from(tile.ready);
                self.cores[c].stats.next_inflight_at_promotion +=
                    usize::from(tile.sent > tile.acks);
                self.cores[c].prefetch_promoted_at = Some(self.now);
            }
            self.log(
                json!({"event":"promote","cycle":self.now,"core":c,"task":next.e,
                "first_ready":next.first.as_ref().is_some_and(|t|t.ready),
                "inflight_bytes":next.first.as_ref().map_or(0,|t|t.sent-t.acks)}),
            );
            self.cores[c].incoming = next.first;
            // Current may wait for input/weights. Only actual issue readiness
            // enables arithmetic; a prediction never enables execution.
            self.reserve_start(next.e, c, false);
            self.last_progress = self.now;
        }
    }

    pub(super) fn admit_next_weights(&mut self) {
        if !self.cfg.next_prefetch || self.now < self.control_free {
            return;
        }
        for off in 0..self.cores.len() {
            let c = (self.rr_desc + off) % self.cores.len();
            let Some(next) = &self.cores[c].next else {
                continue;
            };
            if next.first.is_some() || self.cores[c].free_slots.is_empty() {
                continue;
            }
            assert!(self.cores[c].wslots > self.cfg.group);
            let e = next.e;
            let x = &self.w.experts[e];
            let bytes = x.f.min(4) * align(x.h.min(512) * 2, 32);
            let slot = self.cores[c].free_slots.pop().unwrap();
            assert!(self.cores[c].live_tiles[slot].is_none());
            self.cores[c].live_tiles[slot] = Some(usize::MAX);
            let cost = if self.cfg.control_cost { 2 } else { 0 };
            self.control_free = self.now + cost;
            let tile = Tile {
                n_start: 0,
                k_start: 0,
                nv: x.f.min(4),
                kv: x.h.min(512),
                slot: Some(slot),
                release: self.control_free,
                sent: 0,
                acks: 0,
                bytes,
                ready: false,
                retired: false,
            };
            self.cores[c].next.as_mut().unwrap().first = Some(tile);
            let stats = &mut self.cores[c].stats;
            stats.control_cycles += cost;
            stats.weight_bytes += bytes as u64;
            stats.next_prefetch_tiles += 1;
            stats.next_peak_bytes = 4096;
            let used = self.cores[c].wslots - self.cores[c].free_slots.len();
            self.cores[c].stats.weight_peak_bytes =
                self.cores[c].stats.weight_peak_bytes.max(used * 4096);
            self.log(json!({"event":"next_slot_reserved","cycle":self.now,"task":e,"core":c,"slot":slot,"bytes":bytes}));
            self.last_progress = self.now;
            self.rr_desc = (c + 1) % self.cores.len();
            break;
        }
    }

    fn dma_tile(&self, r: &DmaRequest) -> &Tile {
        let c = &self.cores[r.core];
        let tile = if c.next.as_ref().is_some_and(|n| n.e == r.task) {
            assert_eq!((r.phase, r.tile), (0, 0));
            c.next.as_ref().unwrap().first.as_ref().unwrap()
        } else {
            let s = c.session.as_ref().expect("DMA owner must remain resident");
            assert_eq!((s.e, s.phase), (r.task, r.phase));
            if let Some(run) = &c.run {
                &run.tiles[r.tile]
            } else {
                assert_eq!((r.phase, r.tile), (0, 0));
                c.incoming.as_ref().unwrap()
            }
        };
        assert_eq!(tile.slot, Some(r.slot), "stale DMA slot owner");
        assert!(!tile.retired);
        tile
    }
    fn dma_tile_mut(&mut self, r: &DmaRequest) -> &mut Tile {
        self.dma_tile(r); // Validate immutable identity before selecting mutable view.
        let c = &mut self.cores[r.core];
        if c.next.as_ref().is_some_and(|n| n.e == r.task) {
            c.next.as_mut().unwrap().first.as_mut().unwrap()
        } else if let Some(run) = &mut c.run {
            &mut run.tiles[r.tile]
        } else {
            c.incoming.as_mut().unwrap()
        }
    }
    fn candidate(&self, c: usize) -> Option<DmaRequest> {
        let core = &self.cores[c];
        let mut choice = None;
        if let (Some(s), Some(r)) = (&core.session, &core.run) {
            if r.send_cursor < r.admit {
                choice = Some((s.e, s.phase, r.send_cursor, &r.tiles[r.send_cursor]));
            }
        } else if let (Some(s), Some(t)) = (&core.session, &core.incoming) {
            if t.sent < t.bytes {
                choice = Some((s.e, 0, 0, t));
            }
        }
        if choice.is_none() {
            if let Some(n) = &core.next {
                if let Some(t) = &n.first {
                    if t.sent < t.bytes {
                        choice = Some((n.e, 0, 0, t));
                    }
                }
            }
        }
        let (task, phase, tile, t) = choice?;
        if t.release > self.now || t.sent == t.bytes {
            return None;
        }
        let e = &self.w.experts[task];
        let name = match phase {
            0 => "gate",
            1 => "up",
            4 => "down",
            _ => panic!("non GEMM DMA"),
        };
        let k = if phase < 2 { e.h } else { e.f };
        let base = e.weights[name]["hbm_base"]
            .as_u64()
            .unwrap_or(((task * 3 + if phase < 2 { phase } else { 2 }) * 16 * 1024 * 1024) as u64)
            as usize;
        let stride = e.weights[name]["row_stride_bytes"]
            .as_u64()
            .unwrap_or(align(k * 2, 32) as u64) as usize;
        let tile_stride = align(t.kv * 2, 32);
        Some(DmaRequest {
            serial: self.dma_serial,
            core: c,
            task,
            phase,
            tile,
            slot: t.slot.unwrap(),
            offset: t.sent,
            address: base
                + (t.n_start + t.sent / tile_stride) * stride
                + t.k_start * 2
                + t.sent % tile_stride,
        })
    }

    pub(super) fn issue_runtime(&mut self) {
        let grants = if self.cfg.ideal_hbm {
            self.cfg.credits
        } else {
            self.cfg.hbm_bytes_per_ns / 32
        };
        for _ in 0..grants {
            if self.credit_used >= self.cfg.credits {
                break;
            }
            if self.pending_dma.is_none() {
                let candidates: Vec<_> = (0..self.cores.len())
                    .filter_map(|c| self.candidate(c))
                    .collect();
                for r in &candidates {
                    self.cores[r.core].dma_wait_since.get_or_insert(self.now);
                }
                let pick = candidates.into_iter().min_by_key(|r| {
                    let c = r.core;
                    let age = self.now - self.cores[c].dma_wait_since.unwrap();
                    let ready = self.cores[c].run.as_ref().map_or(0, |run| {
                        self.cores[c]
                            .live_tiles
                            .iter()
                            .flatten()
                            .filter(|&&t| t < run.tiles.len() && run.tiles[t].ready)
                            .count()
                    });
                    (
                        !(age >= 64),
                        if age >= 64 { u64::MAX - age } else { 0 },
                        if self.cfg.arbiter == "urgency" {
                            ready
                        } else {
                            0
                        },
                        (c + self.cores.len() - self.rr_hbm) % self.cores.len(),
                    )
                });
                self.pending_dma = pick;
            }
            let Some(r) = self.pending_dma.clone() else {
                break;
            };
            self.dma_tile(&r); // Reservation persists under DMA backpressure.
            let ready = self.now >= self.cfg.dma_ready_after
                && self.now % self.cfg.dma_ready_period < self.cfg.dma_ready_cycles;
            if !ready {
                self.cores[r.core].stats.dma_backpressure_cycles += 1;
                break; // No AGU advance, charge, new selection or changed tag.
            }
            let t = self.dma_tile_mut(&r);
            assert_eq!(t.sent, r.offset);
            t.sent += 32;
            assert!(t.sent <= t.bytes);
            let complete = t.sent == t.bytes;
            if complete {
                let core = &mut self.cores[r.core];
                if let (Some(s), Some(run)) = (&core.session, &mut core.run) {
                    if s.e == r.task {
                        assert_eq!(run.send_cursor, r.tile);
                        run.send_cursor += 1;
                    }
                }
            }
            self.pending_dma = None;
            assert!(self.outstanding_dma.insert(r.serial, r.clone()).is_none());
            self.dma_serial += 1;
            self.credit_used += 1;
            self.credit_peak = self.credit_peak.max(self.credit_used);
            assert_eq!(self.credit_used, self.outstanding_dma.len());
            self.cores[r.core].stats.dma_accepted += 1;
            let age = self.now - self.cores[r.core].dma_wait_since.take().unwrap_or(self.now);
            self.cores[r.core].stats.eligible_wait_max_cycles =
                self.cores[r.core].stats.eligible_wait_max_cycles.max(age);
            self.rr_hbm = (r.core + 1) % self.cores.len();
            self.log(json!({"event":"dma_fire","cycle":self.now,"request":r}));
            self.last_progress = self.now;
            self.event(
                self.now
                    + if self.cfg.ideal_hbm {
                        0
                    } else {
                        self.cfg.hbm_latency_ns
                    },
                Event::RuntimeReturn(r),
            );
        }
    }
    pub(super) fn return_runtime(&mut self, r: DmaRequest) {
        assert_eq!(self.outstanding_dma.get(&r.serial), Some(&r));
        let t = self.dma_tile(&r);
        let row_stride = align(t.kv * 2, 32);
        let dst = r.slot * 4096 + (r.offset / row_stride) * 1024 + r.offset % row_stride;
        let end = if self.cfg.ideal_onchip {
            self.now
        } else {
            self.cores[r.core]
                .wb
                .access(self.now, &[(dst, 32)], false, false)
        };
        self.event(end, Event::RuntimeAck(r));
    }
    pub(super) fn ack_runtime(&mut self, r: DmaRequest) {
        assert_eq!(self.outstanding_dma.remove(&r.serial), Some(r.clone()));
        let t = self.dma_tile_mut(&r);
        t.acks += 32;
        assert!(t.acks <= t.sent && t.acks <= t.bytes);
        t.ready = t.acks == t.bytes;
        self.credit_used -= 1; // Return SRAM -> reserved W slot has completed.
        self.cores[r.core].stats.dma_landed += 1;
        assert_eq!(self.credit_used, self.outstanding_dma.len());
        self.log(json!({"event":"dma_landed","cycle":self.now,"request":r}));
    }
}
