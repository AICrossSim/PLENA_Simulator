use std::sync::atomic::{AtomicU64, Ordering};

static EXPERIMENTAL_REPORT_CYCLES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn clear_experimental_report_cycles() {
    EXPERIMENTAL_REPORT_CYCLES.store(0, Ordering::Relaxed);
}

pub(crate) fn set_experimental_report_cycles(cycles: u64) {
    EXPERIMENTAL_REPORT_CYCLES.store(cycles.max(1), Ordering::Relaxed);
}

pub(crate) fn experimental_report_cycles() -> Option<u64> {
    match EXPERIMENTAL_REPORT_CYCLES.load(Ordering::Relaxed) {
        0 => None,
        cycles => Some(cycles),
    }
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
    sequence_id: u64,
    remaining_cycles: u64,
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
    next_sequence_id: u64,
    pending_prefetches: Vec<PendingPrefetch>,
    hidden_prefetch_cycles: u64,
    prefetch_ops: u64,
    compute_ops: u64,
    dependent_prefetch_stalls: u64,
}

impl TimingOverlay {
    pub(crate) fn record(&mut self, access: TimingAccess, elapsed_cycles: u64) {
        match access {
            TimingAccess::Prefetch { write_ranges } => {
                if elapsed_cycles == 0 {
                    return;
                }
                let sequence_id = self.next_sequence_id;
                self.next_sequence_id = self.next_sequence_id.wrapping_add(1);
                self.prefetch_ops += 1;
                self.pending_prefetches.push(PendingPrefetch {
                    sequence_id,
                    remaining_cycles: elapsed_cycles,
                    write_ranges,
                });
                self.pending_prefetches
                    .sort_by_key(|prefetch| prefetch.sequence_id);
            }
            TimingAccess::Compute { read_ranges } => {
                self.compute_ops += 1;
                self.retire_dependent_prefetches(&read_ranges);
                self.hide_independent_prefetch_cycles(elapsed_cycles);
            }
            TimingAccess::Barrier => {
                self.pending_prefetches.clear();
            }
            TimingAccess::Other => {}
        }
    }

    pub(crate) fn summary(&self, serial_cycles: u64) -> TimingOverlaySummary {
        let hidden = self.hidden_prefetch_cycles.min(serial_cycles);
        TimingOverlaySummary {
            serial_cycles,
            adjusted_cycles: serial_cycles.saturating_sub(hidden),
            hidden_prefetch_cycles: hidden,
            pending_prefetch_cycles: self.pending_prefetch_cycles(),
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

    fn hide_independent_prefetch_cycles(&mut self, compute_cycles: u64) {
        let mut remaining_compute_cycles = compute_cycles;
        let mut index = 0;
        while remaining_compute_cycles > 0 && index < self.pending_prefetches.len() {
            let hidden = self.pending_prefetches[index]
                .remaining_cycles
                .min(remaining_compute_cycles);
            self.pending_prefetches[index].remaining_cycles -= hidden;
            self.hidden_prefetch_cycles += hidden;
            remaining_compute_cycles -= hidden;
            if self.pending_prefetches[index].remaining_cycles == 0 {
                self.pending_prefetches.remove(index);
            } else {
                index += 1;
            }
        }
    }

    fn pending_prefetch_cycles(&self) -> u64 {
        self.pending_prefetches
            .iter()
            .map(|prefetch| prefetch.remaining_cycles)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{SramRange, SramSpace, TimingAccess, TimingOverlay};

    #[test]
    fn independent_compute_hides_pending_prefetch_cycles() {
        let mut overlay = TimingOverlay::default();
        overlay.record(
            TimingAccess::prefetch(vec![SramRange::new(SramSpace::Vector, 4096, 256)]),
            40,
        );
        overlay.record(
            TimingAccess::compute(vec![SramRange::new(SramSpace::Vector, 0, 64)]),
            6400,
        );
        let summary = overlay.summary(6440);
        assert_eq!(summary.hidden_prefetch_cycles, 40);
        assert_eq!(summary.adjusted_cycles, 6400);
        assert_eq!(summary.dependent_prefetch_stalls, 0);
    }

    #[test]
    fn dependent_compute_keeps_prefetch_cycles_serial() {
        let mut overlay = TimingOverlay::default();
        overlay.record(
            TimingAccess::prefetch(vec![SramRange::new(SramSpace::Vector, 4096, 256)]),
            40,
        );
        overlay.record(
            TimingAccess::compute(vec![SramRange::new(SramSpace::Vector, 4096, 64)]),
            6400,
        );
        let summary = overlay.summary(6440);
        assert_eq!(summary.hidden_prefetch_cycles, 0);
        assert_eq!(summary.adjusted_cycles, 6440);
        assert_eq!(summary.dependent_prefetch_stalls, 1);
    }
}
