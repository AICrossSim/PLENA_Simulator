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
    Prefetch { write_ranges: Vec<SramRange> },
    Compute { read_ranges: Vec<SramRange> },
    Barrier,
    Other,
}

impl TimingAccess {
    pub(crate) fn prefetch(write_ranges: Vec<SramRange>) -> Self {
        Self::Prefetch { write_ranges }
    }

    pub(crate) fn compute(read_ranges: Vec<SramRange>) -> Self {
        Self::Compute { read_ranges }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TimingOverlaySummary {
    pub(crate) serial_cycles: u64,
    pub(crate) adjusted_cycles: u64,
    pub(crate) hidden_prefetch_cycles: u64,
    pub(crate) pending_prefetch_cycles: u64,
    pub(crate) prefetch_ops: u64,
    pub(crate) compute_ops: u64,
    pub(crate) dependent_prefetch_stalls: u64,
}

#[derive(Debug, Default)]
pub(crate) struct TimingOverlay {
    pending_prefetches: Vec<PendingPrefetch>,
    hidden_prefetch_picos: u64,
    prefetch_ops: u64,
    compute_ops: u64,
    dependent_prefetch_stalls: u64,
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
                self.compute_ops += 1;
                self.retire_dependent_prefetches(&read_ranges);
                self.hide_independent_prefetch_picos(elapsed_picos);
            }
            TimingAccess::Barrier => {
                self.pending_prefetches.clear();
            }
            TimingAccess::Other => {}
        }
    }

    pub(crate) fn summary(&self, serial_picos: u64) -> TimingOverlaySummary {
        // Convert to cycles exactly once, here, after accumulating in picoseconds.
        let period = PERIOD.as_picos().max(1);
        let hidden_picos = self.hidden_prefetch_picos.min(serial_picos);
        let serial_cycles = serial_picos.div_ceil(period);
        let adjusted_cycles = serial_picos.saturating_sub(hidden_picos).div_ceil(period);
        TimingOverlaySummary {
            serial_cycles,
            adjusted_cycles,
            // Keep `adjusted == serial - hidden` self-consistent in the cycle domain.
            hidden_prefetch_cycles: serial_cycles.saturating_sub(adjusted_cycles),
            pending_prefetch_cycles: self.pending_prefetch_picos().div_ceil(period),
            prefetch_ops: self.prefetch_ops,
            compute_ops: self.compute_ops,
            dependent_prefetch_stalls: self.dependent_prefetch_stalls,
        }
    }

    fn retire_dependent_prefetches(&mut self, read_ranges: &[SramRange]) {
        let before = self.pending_prefetches.len();
        self.pending_prefetches.retain(|prefetch| {
            !prefetch
                .write_ranges
                .iter()
                .any(|write| read_ranges.iter().any(|read| write.overlaps(*read)))
        });
        self.dependent_prefetch_stalls += (before - self.pending_prefetches.len()) as u64;
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

    #[test]
    fn independent_compute_hides_pending_prefetch_cycles() {
        let mut overlay = TimingOverlay::default();
        overlay.record(
            TimingAccess::prefetch(vec![SramRange::new(SramSpace::Vector, 4096, 256)]),
            40 * NS,
        );
        overlay.record(
            TimingAccess::compute(vec![SramRange::new(SramSpace::Vector, 0, 64)]),
            6400 * NS,
        );
        let summary = overlay.summary(6440 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 40);
        assert_eq!(summary.adjusted_cycles, 6400);
        assert_eq!(summary.dependent_prefetch_stalls, 0);
    }

    #[test]
    fn dependent_compute_keeps_prefetch_cycles_serial() {
        let mut overlay = TimingOverlay::default();
        overlay.record(
            TimingAccess::prefetch(vec![SramRange::new(SramSpace::Vector, 4096, 256)]),
            40 * NS,
        );
        overlay.record(
            TimingAccess::compute(vec![SramRange::new(SramSpace::Vector, 4096, 64)]),
            6400 * NS,
        );
        let summary = overlay.summary(6440 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.adjusted_cycles, 6440);
        assert_eq!(summary.dependent_prefetch_stalls, 1);
    }

    #[test]
    fn sub_cycle_prefetches_hide_in_picosecond_domain_not_per_op() {
        // Two independent 500 ps prefetches total 1000 ps = exactly one cycle.
        // Accumulating per-op div_ceil cycles would round each to a full cycle and
        // over-hide (2 cycles); accumulating in picoseconds hides exactly 1 cycle.
        let mut overlay = TimingOverlay::default();
        overlay.record(
            TimingAccess::prefetch(vec![SramRange::new(SramSpace::Vector, 4096, 256)]),
            500,
        );
        overlay.record(
            TimingAccess::prefetch(vec![SramRange::new(SramSpace::Vector, 8192, 256)]),
            500,
        );
        overlay.record(
            TimingAccess::compute(vec![SramRange::new(SramSpace::Vector, 0, 64)]),
            2 * NS,
        );
        let summary = overlay.summary(3 * NS);
        assert_eq!(summary.hidden_prefetch_cycles, 1);
        assert_eq!(summary.adjusted_cycles, 2);
    }
}
