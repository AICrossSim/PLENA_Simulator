//! Port of the legacy Joint critical-anchor/pair proposal.
//!
//! This module chooses ownership only. The caller supplies the same bounded
//! descriptor window, physical legality and v3 completion estimates used by the
//! other policies, pays all comparison/descriptor service, and revalidates before
//! commit. It does not change prefetch, quota, dataflow, or compute resources.
use std::cmp::Reverse;

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub task: usize,
    pub potential: u8,
    pub eligible: u8,
    pub service: [u64; 2],
    pub finish: [u64; 2],
    pub age: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Placement {
    pub index: usize,
    pub core: usize,
}

#[derive(Debug)]
pub(super) struct Proposal {
    pub anchor: Option<usize>,
    pub age_forced: bool,
    pub placements: Vec<Placement>,
    pub comparisons: u64,
}

pub(super) fn propose(
    candidates: &[Candidate],
    nc: usize,
    age_limit: u8,
    pairing: bool,
    rr: usize,
) -> Proposal {
    assert!((1..=2).contains(&nc));
    assert!(candidates.len() <= 8, "Joint sees the same eight descriptors");
    let min_service = |x: &Candidate| {
        (0..nc)
            .filter(|&c| x.potential & (1 << c) != 0)
            .map(|c| x.service[c])
            .min()
            .unwrap_or(0)
    };
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
    let mut placements = Vec::with_capacity(2);
    let mut comparisons = 0;
    if let Some(a) = anchor {
        let x = &candidates[a];
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
                            [Placement { index: a, core: c }, Placement { index: j, core: other }],
                        ));
                    }
                }
            }
        }
        if let Some((_, pair)) = best_pair {
            placements.extend(pair);
        } else {
            let c = (0..nc)
                .filter(|&c| x.potential & (1 << c) != 0)
                .min_by_key(|&c| (x.finish[c], (c + nc - rr % nc) % nc))
                .unwrap();
            comparisons += x.potential.count_ones() as u64;
            placements.push(Placement { index: a, core: c });
        }
    }
    Proposal { anchor, age_forced: aged.is_some(), placements, comparisons }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(task: usize, service: [u64; 2], finish: [u64; 2], mask: u8) -> Candidate {
        Candidate { task, potential: mask, eligible: mask, service, finish, age: 0 }
    }
    #[test]
    fn critical_anchor_is_distinct_from_earliest_finish() {
        let cs = [c(7, [10, 20], [10, 20], 3), c(9, [100, 200], [100, 200], 3)];
        let p = propose(&cs, 2, 8, true, 0);
        assert_eq!(cs[p.anchor.unwrap()].task, 9);
        assert_eq!(p.placements, vec![Placement { index: 1, core: 0 }, Placement { index: 0, core: 1 }]);
        // A greedy earliest-finish singleton instead chooses task7/core0.
        assert_ne!(p.placements[0], Placement { index: 0, core: 0 });
        assert_eq!(p.comparisons, 2);
    }
    #[test]
    fn legal_pair_has_distinct_cores_and_preserves_aged_anchor() {
        let mut cs = [c(0, [10, 20], [110, 20], 2), c(1, [100, 200], [100, 200], 1)];
        cs[0].age = 8;
        let p = propose(&cs, 2, 8, true, 0);
        assert!(p.age_forced);
        assert_eq!(p.anchor, Some(0));
        assert_eq!(p.placements, vec![Placement { index: 0, core: 1 }, Placement { index: 1, core: 0 }]);
        assert_eq!(p.comparisons, 1);
    }
    #[test]
    fn potential_is_not_irrevocable_eligibility() {
        let mut cs = [c(0, [30, 60], [30, 60], 3)];
        cs[0].eligible = 0;
        let p = propose(&cs, 2, 8, true, 0);
        assert_eq!(p.placements.len(), 1);
        assert_eq!(cs[p.placements[0].index].eligible, 0);
        // The caller must wait/revalidate, not silently bind a busy core.
    }
    #[test]
    fn single_core_uses_critical_anchor_and_empty_window_has_no_grant() {
        let cs = [c(0, [3, 0], [3, 0], 1), c(1, [4, 0], [4, 0], 1)];
        let p = propose(&cs, 1, 8, true, 0);
        assert_eq!(p.placements, vec![Placement { index: 1, core: 0 }]);
        let empty = propose(&[], 2, 8, true, 0);
        assert!(empty.placements.is_empty());
        assert_eq!(empty.comparisons, 0);
    }
}
