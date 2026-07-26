//! Experimental prefetch/compute overlap estimator.
//!
//! `do_ops` executes strictly serially: every opcode is awaited before the next
//! one is decoded. This module does **not** change that. It is post-hoc
//! bookkeeping layered on top of the serial run: it records what each opcode
//! touched and how long it took, then subtracts the prefetch time that a real
//! machine could plausibly have hidden behind later independent compute.
//!
//! # This is not a bound in either direction
//!
//! Do not read `adjusted_cycles` as a floor or a ceiling on real hardware. The
//! model errs in both directions at once:
//!
//! - **Optimistic**: a prefetch whose write range does not overlap a later
//!   compute's read range is hideable up to that compute's own duration (a long
//!   prefetch behind a short compute is only partially hidden). DMA bandwidth,
//!   queue depth, SRAM port contention, and HBM channel contention are not
//!   modelled, and the pending queue is unbounded. Real machines hide less.
//! - **Pessimistic**: `C_BREAK` abandons everything in flight; `Store` retires
//!   its dependencies without hiding anything; scalar and control opcodes
//!   (`TimingAccess::Other`) contribute no hiding capacity at all.
//! - **Incomplete**: only prefetch-write -> compute-read (RAW) is tracked. Writes
//!   performed by compute (`M_*_WO`, the `rd` of `V_*`, `S_MAP_V_FP`) are not, so
//!   WAR and WAW are invisible to the model.
//!
//! What it is good for is *relative* comparison between programs on the same
//! model. Prefetch time that could not be hidden splits across three reported
//! quantities and you need all three: `pending_prefetch_picos` (never consumed),
//! `retired_prefetch_picos` (dropped because a dependent access needed it), and
//! whatever a `Barrier` discarded. Absolute numbers need real concurrent
//! execution plus RTL calibration.

use std::sync::Mutex;

use crate::runtime_config::PERIOD;

/// Overlap-adjusted cycle count handed from `runner::run_from_cli` to `main` for
/// the final "Simulation completed" line. A module-level cell is used rather
/// than a return value because `Executor::spawn` requires `Future<Output = ()>`,
/// so the spawned `run_from_cli` cannot hand this value back to `main` directly.
/// `None` means the experimental overlap model was not active for this run.
static OVERLAP_REPORT_CYCLES: Mutex<Option<u64>> = Mutex::new(None);

pub(crate) fn clear_experimental_report_cycles() {
    *OVERLAP_REPORT_CYCLES.lock().unwrap() = None;
}

pub(crate) fn set_experimental_report_cycles(cycles: u64) {
    *OVERLAP_REPORT_CYCLES.lock().unwrap() = Some(cycles);
}

pub(crate) fn experimental_report_cycles() -> Option<u64> {
    *OVERLAP_REPORT_CYCLES.lock().unwrap()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SramSpace {
    Matrix,
    Vector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SramRange {
    pub(crate) space: SramSpace,
    pub(crate) start: u32,
    pub(crate) len: u32,
}

impl SramRange {
    pub(crate) fn new(space: SramSpace, start: u32, len: u32) -> Self {
        Self { space, start, len }
    }

    fn overlaps(self, other: Self) -> bool {
        if self.space != other.space || self.len == 0 || other.len == 0 {
            return false;
        }
        let self_end = self.start.saturating_add(self.len);
        let other_end = other.start.saturating_add(other.len);
        self.start < other_end && other.start < self_end
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TimingAccess {
    /// HBM -> SRAM fill. Can be hidden behind later independent compute.
    Prefetch { write_ranges: Vec<SramRange> },
    /// Matrix/vector arithmetic. Retires prefetches whose writes overlap
    /// `read_ranges`, and its own duration hides independent pending prefetch
    /// time. `read_ranges` may be empty (`M_BMM_WO` / `M_BMV_WO` overwrite whole
    /// rows and read nothing), in which case the op is pure hiding capacity.
    Compute { read_ranges: Vec<SramRange> },
    /// SRAM -> HBM drain. Retires prefetches whose writes overlap `read_ranges`,
    /// but deliberately does not hide pending prefetch time.
    ///
    /// This is a conservative choice, not a derived one. The model already lets a
    /// prefetch issued before a store survive across it and be hidden by a later
    /// compute, which assumes that prefetch is in flight while the store runs --
    /// so "store and prefetch contend for DMA" would equally forbid *that*.
    /// Without a bandwidth model there is no principled answer, and declining to
    /// credit the store's own duration is the direction that cannot overstate
    /// achievable overlap.
    Store { read_ranges: Vec<SramRange> },
    /// Control barrier: everything in flight is abandoned.
    Barrier,
    /// Does not participate in the overlap model.
    Other,
}

impl TimingAccess {
    pub(crate) fn prefetch(write_ranges: Vec<SramRange>) -> Self {
        Self::Prefetch { write_ranges }
    }

    pub(crate) fn compute(read_ranges: Vec<SramRange>) -> Self {
        Self::Compute { read_ranges }
    }

    pub(crate) fn store(read_ranges: Vec<SramRange>) -> Self {
        Self::Store { read_ranges }
    }
}

#[derive(Clone, Debug)]
struct PendingPrefetch {
    // Duration is accumulated in picoseconds and only converted to cycles once in
    // `summary` — accumulating per-op div_ceil cycles would systematically
    // over-count (two sub-cycle prefetches each rounding up to a full cycle).
    remaining_picos: u64,
    write_ranges: Vec<SramRange>,
}

/// Why a pending prefetch left the queue. Kept separate because a `Barrier`
/// used to drop prefetches without touching any counter, so folding store- and
/// compute-caused retirements into one scalar would silently change what the
/// metric means across releases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetireCause {
    Compute,
    Store,
}

/// Cycle fields are the picosecond fields rounded up to whole `PERIOD`s and are
/// **not** additive: sub-cycle hiding rounds to zero, and each field rounds
/// independently. Do arithmetic on the `_picos` fields. (Same rule the stage
/// profile's schema v3 states; the two are fed from the same `elapsed_picos`.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TimingOverlaySummary {
    pub(crate) serial_picos: u64,
    pub(crate) serial_cycles: u64,
    pub(crate) adjusted_picos: u64,
    pub(crate) adjusted_cycles: u64,
    pub(crate) hidden_prefetch_picos: u64,
    pub(crate) hidden_prefetch_cycles: u64,
    /// Prefetch time still queued at the end of the run: issued, never consumed.
    pub(crate) pending_prefetch_picos: u64,
    pub(crate) pending_prefetch_cycles: u64,
    /// Prefetch time dropped because a dependent access needed the data. Without
    /// this the time vanished from the summary entirely -- it is in neither
    /// `hidden` nor `pending`.
    pub(crate) retired_prefetch_picos: u64,
    /// Prefetch time abandoned by a `Barrier`, likewise otherwise invisible.
    pub(crate) discarded_prefetch_picos: u64,
    pub(crate) prefetch_ops: u64,
    pub(crate) compute_ops: u64,
    pub(crate) store_ops: u64,
    /// Prefetches retired by a dependent compute read.
    pub(crate) compute_prefetch_stalls: u64,
    /// Prefetches retired by a dependent store read. On `main` `H_STORE_V` was a
    /// `Barrier`, which retired silently, so this is reported separately rather
    /// than inflating the compute figure across the change.
    pub(crate) store_prefetch_stalls: u64,
}

#[derive(Debug, Default)]
pub(crate) struct TimingOverlay {
    pending_prefetches: Vec<PendingPrefetch>,
    hidden_prefetch_picos: u64,
    retired_prefetch_picos: u64,
    discarded_prefetch_picos: u64,
    prefetch_ops: u64,
    compute_ops: u64,
    store_ops: u64,
    compute_prefetch_stalls: u64,
    store_prefetch_stalls: u64,
}

impl TimingOverlay {
    pub(crate) fn record(&mut self, access: TimingAccess, elapsed_picos: u64) {
        match access {
            TimingAccess::Prefetch { write_ranges } => {
                if elapsed_picos == 0 {
                    return;
                }
                self.prefetch_ops += 1;
                // Prefetches are only ever appended here; retire/hide remove from the
                // middle but preserve relative order, so pending_prefetches stays in
                // issue order by construction (no ordering key or re-sort needed).
                self.pending_prefetches.push(PendingPrefetch {
                    remaining_picos: elapsed_picos,
                    write_ranges,
                });
            }
            TimingAccess::Compute { read_ranges } => {
                // No elapsed-time guard here. An earlier version skipped zero-duration
                // records on the theory that "no simulator time means no SRAM access",
                // but that premise is false: `Cell::resolve_with` returns without
                // awaiting when the cell is `Ready`, so SRAM reads cost nothing on
                // their own and all opcode duration comes from explicit `cycle!` /
                // `MatrixCore::compute` calls. A latency knob configured to zero would
                // then silently drop a real dependency. Zero-cost no-op arms are
                // mirrored explicitly in `classify_timing_access` instead.
                self.compute_ops += 1;
                self.retire_dependent_prefetches(&read_ranges, RetireCause::Compute);
                self.hide_independent_prefetch_picos(elapsed_picos);
            }
            TimingAccess::Store { read_ranges } => {
                self.store_ops += 1;
                self.retire_dependent_prefetches(&read_ranges, RetireCause::Store);
            }
            TimingAccess::Barrier => {
                self.discarded_prefetch_picos += self.pending_prefetch_picos();
                self.pending_prefetches.clear();
            }
            TimingAccess::Other => {}
        }
    }

    pub(crate) fn summary(&self, serial_picos: u64) -> TimingOverlaySummary {
        // Convert to cycles exactly once, here, after accumulating in picoseconds.
        let period = PERIOD.as_picos().max(1);
        let hidden_picos = self.hidden_prefetch_picos.min(serial_picos);
        let adjusted_picos = serial_picos.saturating_sub(hidden_picos);
        let serial_cycles = serial_picos.div_ceil(period);
        let adjusted_cycles = adjusted_picos.div_ceil(period);
        let pending_picos = self.pending_prefetch_picos();
        TimingOverlaySummary {
            serial_picos,
            serial_cycles,
            adjusted_picos,
            adjusted_cycles,
            hidden_prefetch_picos: hidden_picos,
            // Derived so `adjusted == serial - hidden` stays true in the cycle
            // domain. Note this makes sub-cycle hiding report 0 cycles even though
            // hidden_prefetch_picos is non-zero -- use the picos field.
            hidden_prefetch_cycles: serial_cycles.saturating_sub(adjusted_cycles),
            pending_prefetch_picos: pending_picos,
            pending_prefetch_cycles: pending_picos.div_ceil(period),
            retired_prefetch_picos: self.retired_prefetch_picos,
            discarded_prefetch_picos: self.discarded_prefetch_picos,
            prefetch_ops: self.prefetch_ops,
            compute_ops: self.compute_ops,
            store_ops: self.store_ops,
            compute_prefetch_stalls: self.compute_prefetch_stalls,
            store_prefetch_stalls: self.store_prefetch_stalls,
        }
    }

    fn retire_dependent_prefetches(&mut self, read_ranges: &[SramRange], cause: RetireCause) {
        let mut retired = 0u64;
        let mut retired_picos = 0u64;
        self.pending_prefetches.retain(|prefetch| {
            let dependent = prefetch
                .write_ranges
                .iter()
                .any(|write| read_ranges.iter().any(|read| write.overlaps(*read)));
            if dependent {
                retired += 1;
                // The remaining (not-yet-hidden) time is real prefetch latency that
                // this access had to wait for. Record it so it does not vanish from
                // the summary the way it used to.
                retired_picos += prefetch.remaining_picos;
            }
            !dependent
        });
        self.retired_prefetch_picos += retired_picos;
        match cause {
            RetireCause::Compute => self.compute_prefetch_stalls += retired,
            RetireCause::Store => self.store_prefetch_stalls += retired,
        }
    }

    fn hide_independent_prefetch_picos(&mut self, compute_picos: u64) {
        let mut remaining_compute_picos = compute_picos;
        let mut index = 0;
        while remaining_compute_picos > 0 && index < self.pending_prefetches.len() {
            let hidden = self.pending_prefetches[index]
                .remaining_picos
                .min(remaining_compute_picos);
            self.pending_prefetches[index].remaining_picos -= hidden;
            self.hidden_prefetch_picos += hidden;
            remaining_compute_picos -= hidden;
            if self.pending_prefetches[index].remaining_picos == 0 {
                self.pending_prefetches.remove(index);
            } else {
                index += 1;
            }
        }
    }

    fn pending_prefetch_picos(&self) -> u64 {
        self.pending_prefetches
            .iter()
            .map(|prefetch| prefetch.remaining_picos)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{SramRange, SramSpace, TimingAccess, TimingOverlay};

    // record() and summary() take picoseconds; PERIOD is 1 ns = 1000 ps, so a
    // whole cycle is 1000 ps. Values below are therefore cycles * 1000.
    const NS: u64 = 1000;

    fn vec_range(start: u32, len: u32) -> SramRange {
        SramRange::new(SramSpace::Vector, start, len)
    }

    #[test]
    fn independent_compute_hides_pending_prefetch_cycles() {
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 40 * NS);
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 6400 * NS);
        let summary = overlay.summary(6440 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 40);
        assert_eq!(summary.adjusted_cycles, 6400);
        assert_eq!(summary.compute_prefetch_stalls, 0);
        assert_eq!(summary.retired_prefetch_picos, 0);
    }

    #[test]
    fn dependent_compute_keeps_prefetch_cycles_serial() {
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 40 * NS);
        overlay.record(TimingAccess::compute(vec![vec_range(4096, 64)]), 6400 * NS);
        let summary = overlay.summary(6440 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.adjusted_cycles, 6440);
        assert_eq!(summary.compute_prefetch_stalls, 1);
        // The prefetch time the compute had to wait for is now accounted rather
        // than vanishing from the summary.
        assert_eq!(summary.retired_prefetch_picos, 40 * NS);
    }

    #[test]
    fn sub_cycle_prefetches_hide_in_picosecond_domain_not_per_op() {
        // Two independent 500 ps prefetches total 1000 ps = exactly one cycle.
        // Accumulating per-op div_ceil cycles would round each to a full cycle and
        // over-hide (2 cycles); accumulating in picoseconds hides exactly 1 cycle.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 500);
        overlay.record(TimingAccess::prefetch(vec![vec_range(8192, 256)]), 500);
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 2 * NS);
        let summary = overlay.summary(3 * NS);
        assert_eq!(summary.hidden_prefetch_picos, NS);
        assert_eq!(summary.hidden_prefetch_cycles, 1);
        assert_eq!(summary.adjusted_cycles, 2);
    }

    #[test]
    fn sub_cycle_hiding_is_visible_in_picos_but_rounds_away_in_cycles() {
        // `hidden_prefetch_cycles` is derived as serial - adjusted so the two stay
        // consistent, which means hiding less than one whole period reports zero
        // cycles. The picosecond field is the one that carries the information.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 500);
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 500);
        let summary = overlay.summary(NS);
        assert_eq!(summary.hidden_prefetch_picos, 500);
        assert_eq!(summary.adjusted_picos, 500);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.serial_cycles, 1);
        assert_eq!(summary.adjusted_cycles, 1);
    }

    #[test]
    fn zero_duration_compute_still_retires_its_dependency() {
        // The overlay must not infer "touched nothing" from "took no time": SRAM
        // reads resolve a Ready cell without advancing the clock, so a latency knob
        // set to zero would otherwise drop a real dependency and let the prefetch
        // look hideable.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 40 * NS);
        overlay.record(TimingAccess::compute(vec![vec_range(4096, 64)]), 0);
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 6400 * NS);
        let summary = overlay.summary(6440 * NS);
        assert_eq!(summary.compute_prefetch_stalls, 1);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.retired_prefetch_picos, 40 * NS);
    }

    #[test]
    fn store_retires_its_dependency_without_clearing_unrelated_prefetches() {
        // H_STORE_V used to map to Barrier, which dropped *every* pending prefetch.
        // As a Store it retires only the prefetch whose write range it reads, so the
        // unrelated one survives and later compute can still hide it.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 10 * NS);
        overlay.record(TimingAccess::prefetch(vec![vec_range(8192, 256)]), 30 * NS);
        overlay.record(TimingAccess::store(vec![vec_range(4096, 64)]), 5 * NS);
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 6400 * NS);
        let summary = overlay.summary(6445 * NS);
        assert_eq!(summary.store_ops, 1);
        // Counted against the store, not folded into the compute figure.
        assert_eq!(summary.store_prefetch_stalls, 1);
        assert_eq!(summary.compute_prefetch_stalls, 0);
        assert_eq!(summary.retired_prefetch_picos, 10 * NS);
        // Only the overlapping prefetch retired; the 30 ns one was still hideable.
        assert_eq!(summary.hidden_prefetch_cycles, 30);
    }

    #[test]
    fn store_does_not_hide_prefetch_time_itself() {
        // A store's own duration is deliberately not used as hiding capacity.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 40 * NS);
        overlay.record(TimingAccess::store(vec![vec_range(0, 64)]), 100 * NS);
        let summary = overlay.summary(140 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.adjusted_cycles, 140);
        assert_eq!(summary.pending_prefetch_picos, 40 * NS);
    }

    #[test]
    fn barrier_abandons_everything_in_flight_and_accounts_the_loss() {
        // C_BREAK always costs a cycle, so use a reachable non-zero duration.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 256)]), 40 * NS);
        overlay.record(TimingAccess::Barrier, NS);
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 6400 * NS);
        let summary = overlay.summary(6441 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.pending_prefetch_picos, 0);
        // Previously this 40 ns simply disappeared from every reported field.
        assert_eq!(summary.discarded_prefetch_picos, 40 * NS);
    }

    #[test]
    fn every_issued_prefetch_picosecond_is_accounted_for() {
        // hidden + pending + retired + discarded must equal the total prefetch time
        // issued. This is the invariant the diagnostic fields exist to preserve.
        let mut overlay = TimingOverlay::default();
        overlay.record(TimingAccess::prefetch(vec![vec_range(0, 64)]), 10 * NS);
        overlay.record(TimingAccess::prefetch(vec![vec_range(1024, 64)]), 20 * NS);
        overlay.record(TimingAccess::prefetch(vec![vec_range(2048, 64)]), 30 * NS);
        overlay.record(TimingAccess::prefetch(vec![vec_range(4096, 64)]), 40 * NS);
        // Retire the first by a dependent compute, which also hides 5 ns of others.
        overlay.record(TimingAccess::compute(vec![vec_range(0, 64)]), 5 * NS);
        // Retire the second by a dependent store.
        overlay.record(TimingAccess::store(vec![vec_range(1024, 64)]), NS);
        // Discard whatever is left.
        overlay.record(TimingAccess::Barrier, NS);
        let summary = overlay.summary(107 * NS);
        let accounted = summary.hidden_prefetch_picos
            + summary.pending_prefetch_picos
            + summary.retired_prefetch_picos
            + summary.discarded_prefetch_picos;
        assert_eq!(accounted, (10 + 20 + 30 + 40) * NS);
        assert_eq!(summary.compute_prefetch_stalls, 1);
        assert_eq!(summary.store_prefetch_stalls, 1);
        assert_eq!(summary.prefetch_ops, 4);
    }
}
