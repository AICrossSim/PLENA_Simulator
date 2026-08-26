//! Analytic in-order-issue timing scoreboard.
//!
//! In scoreboard timing mode, `do_ops` still executes every opcode's
//! functional effect serially (values stay bit-identical to a serial run), but
//! the modeled clock is decoupled from execution order. Per instruction:
//!
//! ```text
//! issue  = max(prev_issue + issue_cost,   // in-order 1-op/cycle front end
//!              ready_at(source resources),// RAW
//!              busy_until(unit),          // structural hazard
//!              now)
//! finish = issue + latency                // latency captured, never slept
//! ```
//!
//! then `busy_until(unit) = finish` and every destination's `ready_at` is
//! max-merged with `finish`. The dispatch task sleeps only to `issue` (so DMA
//! launched inside an arm reaches ramulator at its modeled instant) and, at
//! end of program, to `max_finish` — which then *is* the reported total.
//!
//! Hazards this encodes:
//! - **RAW** via `ready_at` on reads.
//! - **WAW** via max-merge at commit: a reader after two writers waits for
//!   both finishes, which is the instant the location is last stable.
//! - **WAR** needs no stall: reads are modeled at issue, and in-order issue
//!   means a later writer can never be modeled before an earlier reader's
//!   issue. (A write landing between a reader's issue and its internal
//!   operand fetch is sub-cycle detail outside this model.)
//! - Units are non-pipelined: back-to-back ops on one unit serialize on
//!   `busy_until` (a systolic array is busy for its whole feed time).
//!
//! Asynchronous DMA (prefetches) is carried separately as [`PendingDma`]:
//! its destination `ready_at` is unknown at issue, so the entry parks the
//! completion channel and the first dependent instruction awaits it, feeding
//! the *real* completion instant back into the range tracker.

use runtime::{Duration, Instant};
use tokio::sync::oneshot;

use super::access::{AccumKind, Cfg, OpAccess, Resource, SramRange, SramSpace, Unit};
use crate::runtime_config::PERIOD;

/// One contiguous SRAM extent with the instant its last writer finishes.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: u32,
    /// Exclusive.
    end: u32,
    at: Instant,
}

/// Per-space write-timestamp map over element-address intervals.
///
/// Spans are sorted, non-overlapping, and capped at `MAX_SPANS`: beyond that
/// the two closest spans merge, taking the max instant over their union —
/// strictly conservative (coarsening can only add stalls, never lose one).
#[derive(Debug, Default)]
pub(crate) struct RangeTracker {
    spans: Vec<Span>,
}

const MAX_SPANS: usize = 64;

impl RangeTracker {
    /// Latest finish instant of any writer overlapping `[start, start+len)`;
    /// `INIT` when the range has no tracked writer.
    fn ready_at(&self, start: u32, len: u32) -> Instant {
        let end = start.saturating_add(len);
        self.spans
            .iter()
            .filter(|s| s.start < end && start < s.end)
            .map(|s| s.at)
            .max()
            .unwrap_or(Instant::INIT)
    }

    /// Record a write to `[start, start+len)` finishing at `at`, max-merging
    /// with existing overlapping spans (WAW keeps the later stable instant).
    fn record_write(&mut self, start: u32, len: u32, at: Instant) {
        if len == 0 {
            return;
        }
        let end = start.saturating_add(len);
        let mut result: Vec<Span> = Vec::with_capacity(self.spans.len() + 2);
        let mut cursor = start;
        for s in self.spans.drain(..) {
            if s.end <= start || s.start >= end {
                result.push(s);
                continue;
            }
            // Portion of the old span before the new range keeps its instant.
            if s.start < start {
                result.push(Span {
                    start: s.start,
                    end: start,
                    at: s.at,
                });
            }
            let ov_start = s.start.max(start);
            let ov_end = s.end.min(end);
            // Portion of the new range before this overlap is new-only.
            if cursor < ov_start {
                result.push(Span {
                    start: cursor,
                    end: ov_start,
                    at,
                });
            }
            result.push(Span {
                start: ov_start,
                end: ov_end,
                at: s.at.max(at),
            });
            cursor = ov_end.max(cursor);
            // Portion of the old span after the new range keeps its instant.
            if s.end > end {
                result.push(Span {
                    start: end,
                    end: s.end,
                    at: s.at,
                });
            }
        }
        if cursor < end {
            result.push(Span {
                start: cursor,
                end,
                at,
            });
        }
        result.sort_by_key(|s| s.start);
        // Merge touching spans with identical instants.
        let mut merged: Vec<Span> = Vec::with_capacity(result.len());
        for s in result {
            match merged.last_mut() {
                Some(last) if last.end == s.start && last.at == s.at => last.end = s.end,
                _ => merged.push(s),
            }
        }
        self.spans = merged;
        self.coarsen();
    }

    fn coarsen(&mut self) {
        while self.spans.len() > MAX_SPANS {
            let mut best = 0;
            let mut best_gap = u32::MAX;
            for i in 0..self.spans.len() - 1 {
                let gap = self.spans[i + 1].start - self.spans[i].end;
                if gap < best_gap {
                    best_gap = gap;
                    best = i;
                }
            }
            let right = self.spans.remove(best + 1);
            let left = &mut self.spans[best];
            left.end = right.end;
            left.at = left.at.max(right.at);
        }
    }
}

/// Direction of an in-flight asynchronous DMA transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DmaKind {
    /// HBM → SRAM fill (`H_PREFETCH_*`): `writes` are its SRAM destination.
    Prefetch,
    /// SRAM → HBM drain (`H_STORE_V`): the vram rows were snapshotted at
    /// issue, so it carries no SRAM writes — only its HBM-side effect, which
    /// is ordered conservatively (any later `H_*` op waits for it).
    Store,
}

/// An issued-but-uncompleted asynchronous DMA transfer.
///
/// The completion channel carries the transfer's real finish instant, sent by
/// the spawned completer task. The first later instruction whose access
/// overlaps `writes` awaits it (advancing virtual time through the memory
/// model) and feeds the instant back via [`Scoreboard::retire_dma`].
#[derive(Debug)]
pub(crate) struct PendingDma {
    pub(crate) kind: DmaKind,
    pub(crate) writes: Vec<SramRange>,
    pub(crate) done: oneshot::Receiver<Instant>,
}

/// Stall/busy accounting reported at end of run.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ScoreboardStats {
    pub(crate) ops: u64,
    /// Extra issue delay attributed to RAW dependencies (picoseconds).
    pub(crate) data_stall_picos: u64,
    /// Extra issue delay attributed to a busy functional unit (picoseconds).
    pub(crate) structural_stall_picos: u64,
    /// Time dispatch spent awaiting asynchronous DMA completions.
    pub(crate) dma_wait_picos: u64,
    /// Modeled occupancy per unit (sum of latencies; overlaps not collapsed).
    pub(crate) unit_busy_picos: [u64; Unit::COUNT],
    pub(crate) unit_ops: [u64; Unit::COUNT],
}

pub(crate) struct Scoreboard {
    /// Issue instant of the previous instruction (front-end pacing).
    prev_issue: Option<Instant>,
    /// Finish instant of the previous instruction (serialize mode only).
    prev_finish: Instant,
    /// Validation mode: force fully serial issue (must reproduce serial-mode
    /// cycle counts exactly).
    serialize: bool,
    issue_cost: Duration,
    unit_busy_until: [Instant; Unit::COUNT],
    gp_ready: [Instant; 16],
    fp_ready: [Instant; 8],
    hbm_ready: [Instant; 16],
    cfg_ready: [Instant; Cfg::COUNT],
    accum_ready: [Instant; AccumKind::COUNT],
    sram: [RangeTracker; SramSpace::COUNT],
    pending_dma: Vec<PendingDma>,
    max_finish: Instant,
    /// Optional JSONL per-op trace sink (`--scoreboard-trace`).
    trace: Option<Box<dyn std::io::Write + Send>>,
    pub(crate) stats: ScoreboardStats,
}

impl Scoreboard {
    pub(crate) fn new(serialize: bool) -> Self {
        Self {
            prev_issue: None,
            prev_finish: Instant::INIT,
            serialize,
            issue_cost: PERIOD,
            unit_busy_until: [Instant::INIT; Unit::COUNT],
            gp_ready: [Instant::INIT; 16],
            fp_ready: [Instant::INIT; 8],
            hbm_ready: [Instant::INIT; 16],
            cfg_ready: [Instant::INIT; Cfg::COUNT],
            accum_ready: [Instant::INIT; AccumKind::COUNT],
            sram: Default::default(),
            pending_dma: Vec::new(),
            max_finish: Instant::INIT,
            trace: None,
            stats: ScoreboardStats::default(),
        }
    }

    /// Whether serial-equivalence validation mode is active. Dispatch also
    /// keeps prefetches inline in this mode, so the run must reproduce serial
    /// timing exactly.
    pub(crate) fn is_serialize(&self) -> bool {
        self.serialize
    }

    pub(crate) fn set_trace(&mut self, sink: Box<dyn std::io::Write + Send>) {
        self.trace = Some(sink);
    }

    /// Emit one JSONL trace line for a committed op (no-op unless a trace
    /// sink is installed).
    pub(crate) fn trace_op(
        &mut self,
        pc: usize,
        op: &dyn std::fmt::Debug,
        unit: Unit,
        issue: Instant,
        finish: Instant,
    ) {
        if let Some(sink) = self.trace.as_mut() {
            let issue_ps = (issue - Instant::INIT).as_picos();
            let finish_ps = (finish - Instant::INIT).as_picos();
            let op_name = format!("{op:?}");
            let _ = writeln!(
                sink,
                "{{\"pc\":{pc},\"op\":{:?},\"unit\":\"{}\",\"issue_ps\":{issue_ps},\"finish_ps\":{finish_ps}}}",
                op_name,
                unit.name(),
            );
        }
    }

    fn resource_ready(&self, r: &Resource) -> Instant {
        match r {
            Resource::Gp(i) => self.gp_ready[*i as usize],
            Resource::Fp(i) => self.fp_ready[*i as usize],
            Resource::Hbm(i) => self.hbm_ready[*i as usize],
            Resource::Cfg(c) => self.cfg_ready[c.index()],
            Resource::Accum(a) => self.accum_ready[a.index()],
            Resource::Sram(range) => {
                self.sram[range.space.index()].ready_at(range.start, range.len)
            }
        }
    }

    fn set_resource_ready(&mut self, r: &Resource, at: Instant) {
        match r {
            Resource::Gp(i) => {
                let slot = &mut self.gp_ready[*i as usize];
                *slot = (*slot).max(at);
            }
            Resource::Fp(i) => {
                let slot = &mut self.fp_ready[*i as usize];
                *slot = (*slot).max(at);
            }
            Resource::Hbm(i) => {
                let slot = &mut self.hbm_ready[*i as usize];
                *slot = (*slot).max(at);
            }
            Resource::Cfg(c) => {
                let slot = &mut self.cfg_ready[c.index()];
                *slot = (*slot).max(at);
            }
            Resource::Accum(a) => {
                let slot = &mut self.accum_ready[a.index()];
                *slot = (*slot).max(at);
            }
            Resource::Sram(range) => {
                self.sram[range.space.index()].record_write(range.start, range.len, at);
            }
        }
    }

    /// Earliest instant `access` may issue, given the current virtual time.
    /// Also attributes any stall beyond the front-end floor to its cause.
    pub(crate) fn issue_bound(&mut self, access: &OpAccess, now: Instant) -> Instant {
        let front_end = match self.prev_issue {
            Some(prev) => now.max(prev + self.issue_cost),
            None => now,
        };
        let structural = self.unit_busy_until[access.unit.index()];
        let data = access
            .reads
            .iter()
            .map(|r| self.resource_ready(r))
            .max()
            .unwrap_or(Instant::INIT);
        let mut bound = front_end.max(structural).max(data);
        if access.barrier {
            bound = bound.max(self.max_finish);
        }
        if self.serialize {
            bound = bound.max(self.prev_finish);
        }
        if bound > front_end && !access.barrier && !self.serialize {
            let stall = (bound - front_end).as_picos();
            if data >= structural {
                self.stats.data_stall_picos += stall;
            } else {
                self.stats.structural_stall_picos += stall;
            }
        }
        bound
    }

    /// Record an instruction's issue and captured latency; returns its finish
    /// instant. Destinations become ready (max-merged) at that finish.
    pub(crate) fn commit(
        &mut self,
        access: &OpAccess,
        issue: Instant,
        latency: Duration,
    ) -> Instant {
        let finish = issue + latency;
        self.prev_issue = Some(issue);
        self.prev_finish = self.prev_finish.max(finish);
        let unit = access.unit.index();
        self.unit_busy_until[unit] = self.unit_busy_until[unit].max(finish);
        self.stats.unit_busy_picos[unit] += latency.as_picos();
        self.stats.unit_ops[unit] += 1;
        self.stats.ops += 1;
        for w in &access.writes {
            self.set_resource_ready(w, finish);
        }
        self.max_finish = self.max_finish.max(finish);
        finish
    }

    /// Park an in-flight asynchronous DMA whose SRAM destinations become
    /// ready only when its completion channel fires.
    pub(crate) fn register_dma(
        &mut self,
        kind: DmaKind,
        writes: Vec<SramRange>,
        done: oneshot::Receiver<Instant>,
    ) {
        self.pending_dma.push(PendingDma { kind, writes, done });
    }

    /// Remove and return the pending DMAs whose destinations overlap any SRAM
    /// range `access` reads or writes (RAW and WAW ordering on the fill).
    /// Stores carry no SRAM writes (their rows were snapshotted at issue), so
    /// they are never taken here.
    pub(crate) fn take_overlapping_dma(&mut self, access: &OpAccess) -> Vec<PendingDma> {
        let ranges: Vec<SramRange> = access
            .reads
            .iter()
            .chain(access.writes.iter())
            .filter_map(|r| match r {
                Resource::Sram(range) => Some(*range),
                _ => None,
            })
            .collect();
        if ranges.is_empty() {
            return Vec::new();
        }
        self.take_dma_where(|dma| {
            dma.writes
                .iter()
                .any(|w| ranges.iter().any(|r| w.overlaps(*r)))
        })
    }

    /// Drain policy for a new `H_PREFETCH_*`: every outstanding store (its
    /// HBM writes may cover the region this prefetch reads — no HBM range
    /// tracking, so order conservatively) plus any prefetch overlapping the
    /// new SRAM destination (WAW on the fill).
    pub(crate) fn take_dma_for_prefetch(&mut self, access: &OpAccess) -> Vec<PendingDma> {
        let mut taken = self.take_dma_where(|dma| dma.kind == DmaKind::Store);
        taken.extend(self.take_overlapping_dma(access));
        taken
    }

    fn take_dma_where(&mut self, mut pred: impl FnMut(&PendingDma) -> bool) -> Vec<PendingDma> {
        let mut taken = Vec::new();
        let mut kept = Vec::new();
        for dma in self.pending_dma.drain(..) {
            if pred(&dma) {
                taken.push(dma);
            } else {
                kept.push(dma);
            }
        }
        self.pending_dma = kept;
        taken
    }

    /// Remove and return every pending DMA (barriers, stores, end of program).
    pub(crate) fn take_all_dma(&mut self) -> Vec<PendingDma> {
        std::mem::take(&mut self.pending_dma)
    }

    /// Feed a completed DMA's real finish instant back into the trackers.
    pub(crate) fn retire_dma(&mut self, writes: &[SramRange], completed_at: Instant) {
        for w in writes {
            self.sram[w.space.index()].record_write(w.start, w.len, completed_at);
        }
        self.max_finish = self.max_finish.max(completed_at);
    }

    pub(crate) fn note_dma_wait(&mut self, waited: Duration) {
        self.stats.dma_wait_picos += waited.as_picos();
    }

    pub(crate) fn max_finish(&self) -> Instant {
        self.max_finish
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::access::{OpAccess, Resource, SramRange, SramSpace, Unit};

    fn ns(n: u64) -> Instant {
        Instant::INIT + Duration::from_nanos(n)
    }

    fn cycles(n: u32) -> Duration {
        PERIOD * n
    }

    fn compute(unit: Unit, reads: Vec<Resource>, writes: Vec<Resource>) -> OpAccess {
        OpAccess {
            unit,
            barrier: false,
            reads,
            writes,
        }
    }

    fn vrange(start: u32, len: u32) -> SramRange {
        SramRange::new(SramSpace::Vector, start, len)
    }

    #[test]
    fn independent_ops_on_different_units_overlap() {
        let mut sb = Scoreboard::new(false);
        let now = Instant::INIT;

        // Long matrix op at cycle 0.
        let m = compute(Unit::Matrix, vec![], vec![]);
        let issue = sb.issue_bound(&m, now);
        assert_eq!(issue, now);
        let m_finish = sb.commit(&m, issue, cycles(100));
        assert_eq!(m_finish, ns(100));

        // Independent scalar op issues 1 cycle later, hidden under the matrix.
        let s = compute(Unit::Scalar, vec![Resource::Gp(1)], vec![Resource::Gp(2)]);
        let issue = sb.issue_bound(&s, now);
        assert_eq!(issue, ns(1), "front end paces 1 op/cycle");
        let s_finish = sb.commit(&s, issue, cycles(1));
        assert_eq!(s_finish, ns(2));

        assert_eq!(sb.max_finish(), ns(100), "scalar work is fully hidden");
    }

    #[test]
    fn raw_register_dependency_stalls_the_reader() {
        let mut sb = Scoreboard::new(false);
        let now = Instant::INIT;

        let producer = compute(Unit::Vector, vec![], vec![Resource::Fp(3)]);
        let issue = sb.issue_bound(&producer, now);
        sb.commit(&producer, issue, cycles(20));

        let consumer = compute(Unit::Scalar, vec![Resource::Fp(3)], vec![]);
        let issue = sb.issue_bound(&consumer, now);
        assert_eq!(issue, ns(20), "reader waits for the fp write to finish");
        assert!(sb.stats.data_stall_picos > 0);
    }

    #[test]
    fn same_unit_ops_serialize_structurally() {
        let mut sb = Scoreboard::new(false);
        let now = Instant::INIT;

        let a = compute(Unit::Matrix, vec![], vec![]);
        let issue = sb.issue_bound(&a, now);
        sb.commit(&a, issue, cycles(64));

        let b = compute(Unit::Matrix, vec![], vec![]);
        let issue = sb.issue_bound(&b, now);
        assert_eq!(issue, ns(64), "single matrix core is busy");
        let finish = sb.commit(&b, issue, cycles(64));
        assert_eq!(finish, ns(128));
        assert!(sb.stats.structural_stall_picos > 0);
    }

    #[test]
    fn sram_range_dependency_uses_overlap_not_equality() {
        let mut sb = Scoreboard::new(false);
        let now = Instant::INIT;

        let writer = compute(
            Unit::Vector,
            vec![],
            vec![Resource::Sram(vrange(1024, 256))],
        );
        let issue = sb.issue_bound(&writer, now);
        sb.commit(&writer, issue, cycles(10));

        // Overlapping read stalls.
        let reader = compute(Unit::Scalar, vec![Resource::Sram(vrange(1152, 64))], vec![]);
        assert_eq!(sb.issue_bound(&reader, now), ns(10));

        // Disjoint read does not.
        let disjoint = compute(Unit::Scalar, vec![Resource::Sram(vrange(4096, 64))], vec![]);
        assert_eq!(sb.issue_bound(&disjoint, now), ns(1));
    }

    #[test]
    fn waw_keeps_the_latest_stable_instant() {
        let mut sb = Scoreboard::new(false);
        let now = Instant::INIT;

        // Long writer first, short writer second (finishes earlier).
        let w1 = compute(Unit::Matrix, vec![], vec![Resource::Sram(vrange(0, 64))]);
        let issue = sb.issue_bound(&w1, now);
        sb.commit(&w1, issue, cycles(100));

        let w2 = compute(Unit::Vector, vec![], vec![Resource::Sram(vrange(0, 64))]);
        let issue = sb.issue_bound(&w2, now);
        sb.commit(&w2, issue, cycles(1));

        // A reader must wait for the *later* finish (max-merge), not w2's.
        let r = compute(Unit::Scalar, vec![Resource::Sram(vrange(0, 64))], vec![]);
        assert_eq!(sb.issue_bound(&r, now), ns(100));
    }

    #[test]
    fn barrier_waits_for_everything_in_flight() {
        let mut sb = Scoreboard::new(false);
        let now = Instant::INIT;

        let long = compute(Unit::Matrix, vec![], vec![]);
        let issue = sb.issue_bound(&long, now);
        sb.commit(&long, issue, cycles(500));

        let mut barrier = compute(Unit::Scalar, vec![], vec![]);
        barrier.barrier = true;
        assert_eq!(sb.issue_bound(&barrier, now), ns(500));
    }

    #[test]
    fn serialize_mode_reproduces_the_serial_sum() {
        let mut sb = Scoreboard::new(true);
        let now = Instant::INIT;
        let latencies = [7u32, 1, 20, 3];
        let mut finish = Instant::INIT;
        for lat in latencies {
            let a = compute(Unit::Scalar, vec![], vec![]);
            let issue = sb.issue_bound(&a, now);
            assert_eq!(issue, finish, "serialize forces issue at previous finish");
            finish = sb.commit(&a, issue, cycles(lat));
        }
        assert_eq!(finish, ns(31));
        assert_eq!(sb.max_finish(), ns(31));
    }

    #[test]
    fn pending_dma_is_taken_only_by_overlapping_accesses() {
        let mut sb = Scoreboard::new(false);
        let (_tx, rx) = oneshot::channel();
        sb.register_dma(DmaKind::Prefetch, vec![vrange(4096, 256)], rx);

        let disjoint = compute(Unit::Vector, vec![Resource::Sram(vrange(0, 64))], vec![]);
        assert!(sb.take_overlapping_dma(&disjoint).is_empty());

        let dependent = compute(Unit::Vector, vec![Resource::Sram(vrange(4100, 4))], vec![]);
        let taken = sb.take_overlapping_dma(&dependent);
        assert_eq!(taken.len(), 1);
        assert!(sb.take_all_dma().is_empty());
    }

    #[test]
    fn pending_stores_are_drained_by_prefetches_but_not_by_compute() {
        let mut sb = Scoreboard::new(false);
        let (_tx, rx) = oneshot::channel();
        // A store carries no SRAM writes (rows snapshotted at issue).
        sb.register_dma(DmaKind::Store, vec![], rx);

        // Compute ops never wait on stores, whatever they touch.
        let compute_access = compute(Unit::Vector, vec![Resource::Sram(vrange(0, 64))], vec![]);
        assert!(sb.take_overlapping_dma(&compute_access).is_empty());

        // A new prefetch conservatively drains every outstanding store (its
        // HBM source may overlap the store's HBM destination).
        let prefetch_access = compute(Unit::Dma, vec![], vec![Resource::Sram(vrange(4096, 256))]);
        assert_eq!(sb.take_dma_for_prefetch(&prefetch_access).len(), 1);
        assert!(sb.take_all_dma().is_empty());
    }

    #[test]
    fn retired_dma_feeds_the_range_tracker() {
        let mut sb = Scoreboard::new(false);
        sb.retire_dma(&[vrange(4096, 256)], ns(777));
        let reader = compute(Unit::Vector, vec![Resource::Sram(vrange(4096, 64))], vec![]);
        assert_eq!(sb.issue_bound(&reader, Instant::INIT), ns(777));
        assert_eq!(sb.max_finish(), ns(777));
    }

    #[test]
    fn range_tracker_splits_and_max_merges() {
        let mut t = RangeTracker::default();
        t.record_write(0, 100, ns(50));
        t.record_write(40, 20, ns(10)); // earlier writer over a sub-range: max wins
        assert_eq!(t.ready_at(45, 1), ns(50));
        t.record_write(40, 20, ns(90));
        assert_eq!(t.ready_at(45, 1), ns(90));
        assert_eq!(t.ready_at(0, 10), ns(50));
        assert_eq!(t.ready_at(200, 10), Instant::INIT);
    }

    #[test]
    fn range_tracker_coarsening_is_conservative() {
        let mut t = RangeTracker::default();
        // Far more disjoint spans than the cap, with increasing instants.
        for i in 0..(super::MAX_SPANS as u32 * 4) {
            t.record_write(i * 100, 10, ns(i as u64 + 1));
        }
        assert!(t.spans.len() <= super::MAX_SPANS);
        // Every original range still reports an instant >= what was written.
        for i in 0..(super::MAX_SPANS as u32 * 4) {
            assert!(
                t.ready_at(i * 100, 10) >= ns(i as u64 + 1),
                "coarsening lost a hazard at span {i}"
            );
        }
    }
}
