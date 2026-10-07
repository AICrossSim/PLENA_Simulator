//! Bounded resident-weight-group traversal. DMA descriptor order is unchanged.
//! The compiled group length fits in the existing 32-byte descriptor. Runtime
//! control uses a four-register sequencer and per-slot sequence/group tags.
use super::Band;
use std::collections::VecDeque;

#[derive(Debug)]
pub(super) struct TilePlan {
    pub band: usize,
    pub kt: usize,
    pub sequence: usize,
    pub group_len: usize,
}

pub(super) fn group_tiles(
    order: VecDeque<(usize, usize)>,
    limit: usize,
    bands: &[Band],
    jobs: &[crate::moe_spatial::Job],
    m_lanes: usize,
    x_slots: usize,
) -> VecDeque<TilePlan> {
    let order: Vec<_> = order.into_iter().collect();
    let mut out = VecDeque::new();
    let mut start = 0;
    while start < order.len() {
        let (first, kt) = order[start];
        // When the complete X working set already fits, preserve consecutive
        // M blocks and the W gather reuse. No measured latency guides this rule.
        let group_limit = if jobs[bands[first].job].m.div_ceil(m_lanes) <= x_slots {
            1
        } else {
            limit.max(1)
        };
        let len = order[start..]
            .iter()
            .take(group_limit)
            .take_while(|&&(band, k)| bands[band].job == bands[first].job && k == kt)
            .count();
        for (offset, &(band, kt)) in order[start..start + len].iter().enumerate() {
            out.push_back(TilePlan {
                band,
                kt,
                sequence: start + offset,
                group_len: len,
            });
        }
        start += len;
    }
    out
}

#[derive(Clone, Debug, Default)]
pub(super) struct ReuseCursor {
    base: usize,
    offset: usize,
    pub row: usize,
    group_len: usize,
}

impl ReuseCursor {
    pub fn target_sequence(&self) -> usize {
        self.base + self.offset
    }

    pub fn advance(&mut self, group_len: usize, rows: usize, total_rows: usize) {
        if self.offset == 0 {
            self.group_len = group_len;
        }
        assert_eq!(self.group_len, group_len);
        self.offset += 1;
        if self.offset == group_len {
            self.offset = 0;
            self.row += rows;
            if self.row == total_rows {
                self.base += group_len;
                self.row = 0;
            }
        }
    }
}
