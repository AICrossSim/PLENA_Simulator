//! Projection plans for the fixed BF16 M x 4 x 512 datapath.
//!
//! A `Group` is one resident set of at most `group_n` weight tiles for one
//! K segment. Groups are ordered by N-group, then increasing K. Issues within
//! each group are ordered by M block, then N band: one staged X block can serve
//! every resident weight tile before X is replaced. The plan never splits one
//! output's K reduction across owners or combines rows from different experts.
//!
//! These are work/traffic counts, not a cycle model. In particular, issue count
//! times pipeline latency is not a valid elapsed-time estimate.

use std::fmt;

pub const N_TILE: usize = 4;
pub const K_TILE: usize = 512;
pub const BF16_BYTES: usize = 2;
pub const FP32_BYTES: usize = 4;
pub const TRANSACTION_BYTES: usize = 32;
pub const OUTPUT_ROW_DATA_BYTES: usize = N_TILE * FP32_BYTES;
pub const OUTPUT_ROW_METADATA_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection {
    pub m: usize,
    /// Number of output columns owned by this projection, starting at n_start.
    pub n: usize,
    pub k: usize,
    /// Absolute output-column index in the expert's weight/output tensors.
    pub n_start: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionKind {
    Gate,
    Up,
    Down,
}

/// Logical GEMMs for a gated FFN. Activation and gate/up multiplication are
/// explicit dependencies for the caller, not extra work hidden in these plans.
pub fn ffn_projections(
    m: usize,
    hidden: usize,
    intermediate: usize,
) -> [(ProjectionKind, Projection); 3] {
    [
        (
            ProjectionKind::Gate,
            Projection {
                m,
                n: intermediate,
                k: hidden,
                n_start: 0,
            },
        ),
        (
            ProjectionKind::Up,
            Projection {
                m,
                n: intermediate,
                k: hidden,
                n_start: 0,
            },
        ),
        (
            ProjectionKind::Down,
            Projection {
                m,
                n: hidden,
                k: intermediate,
                n_start: 0,
            },
        ),
    ]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightTile {
    pub n_start: usize,
    pub n_valid: usize,
    pub k_start: usize,
    pub k_valid: usize,
    /// Useful BF16 payload, without transaction padding.
    pub logical_bytes: u64,
    /// W[N,K] rows start on 32-byte boundaries. Each physical row is rounded
    /// independently; padding is not pooled across adjacent output rows.
    pub native_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub m_start: usize,
    pub m_valid: usize,
    pub n_start: usize,
    pub n_valid: usize,
    pub k_start: usize,
    pub k_valid: usize,
}

impl Issue {
    pub fn useful_macs(&self) -> u64 {
        self.m_valid as u64 * self.n_valid as u64 * self.k_valid as u64
    }

    /// Physical allocation, including M/N/K padding; not extra useful math.
    pub fn issued_macs(&self, m_core: usize) -> u64 {
        m_core as u64 * N_TILE as u64 * K_TILE as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub id: usize,
    pub n_start: usize,
    /// Total real columns covered by this resident group (not tile count).
    pub n_valid: usize,
    pub k_start: usize,
    pub k_valid: usize,
    pub tiles: Vec<WeightTile>,
    /// Sorted by m_start, then n_start. The caller may stage X once for each
    /// distinct (phase, k_start, m_start) in this group.
    pub issues: Vec<Issue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    ZeroCoreRows,
    ZeroGroupWidth,
    ZeroK,
    DimensionOverflow,
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCoreRows => write!(f, "m_core must be positive"),
            Self::ZeroGroupWidth => write!(f, "group_n must be positive"),
            Self::ZeroK => write!(f, "nonempty projections require k > 0"),
            Self::DimensionOverflow => {
                write!(f, "projection dimensions overflow address or byte counts")
            }
        }
    }
}

impl std::error::Error for PlanError {}

fn ceil_div(n: usize, d: usize) -> usize {
    n / d + usize::from(n % d != 0)
}

fn validate(p: Projection, m_core: usize, group_n: usize) -> Result<(), PlanError> {
    if m_core == 0 {
        return Err(PlanError::ZeroCoreRows);
    }
    if group_n == 0 {
        return Err(PlanError::ZeroGroupWidth);
    }
    if p.m != 0 && p.n != 0 && p.k == 0 {
        return Err(PlanError::ZeroK);
    }
    p.n_start
        .checked_add(p.n)
        .ok_or(PlanError::DimensionOverflow)?;
    group_n
        .checked_mul(N_TILE)
        .ok_or(PlanError::DimensionOverflow)?;
    // Conservative bound on padded operand/accumulator accounting. The
    // multiplication stays checked even for adversarial, unallocatable inputs.
    let issues = (ceil_div(p.m, m_core) as u128)
        .checked_mul(ceil_div(p.n, N_TILE) as u128)
        .and_then(|v| v.checked_mul(ceil_div(p.k, K_TILE) as u128))
        .ok_or(PlanError::DimensionOverflow)?;
    let padded = issues
        .checked_mul(m_core as u128)
        .and_then(|v| v.checked_mul(N_TILE as u128 * K_TILE as u128))
        .and_then(|v| v.checked_mul(32))
        .ok_or(PlanError::DimensionOverflow)?;
    if padded > u64::MAX as u128 {
        return Err(PlanError::DimensionOverflow);
    }
    Ok(())
}

/// Builds work only. The caller must check simultaneous SRAM occupancy,
/// reserve output/weight/return slots, and charge ports/control before issuing.
/// Any positive G is representable; the experiment scans G=1,2,4 and the caller
/// enforces the actual per-core weight-slot budget.
pub fn build_groups(p: Projection, m_core: usize, group_n: usize) -> Result<Vec<Group>, PlanError> {
    validate(p, m_core, group_n)?;
    let mut groups = Vec::new();
    if p.m == 0 || p.n == 0 {
        return Ok(groups);
    }
    let group_columns = group_n * N_TILE;
    for n_offset in (0..p.n).step_by(group_columns) {
        let n_valid = (p.n - n_offset).min(group_columns);
        for k_start in (0..p.k).step_by(K_TILE) {
            let k_valid = (p.k - k_start).min(K_TILE);
            let mut tiles = Vec::new();
            for local_n in (0..n_valid).step_by(N_TILE) {
                let tile_n = (n_valid - local_n).min(N_TILE);
                let payload_per_row = k_valid * BF16_BYTES;
                tiles.push(WeightTile {
                    n_start: p.n_start + n_offset + local_n,
                    n_valid: tile_n,
                    k_start,
                    k_valid,
                    logical_bytes: (tile_n * payload_per_row) as u64,
                    native_bytes: (tile_n
                        * ceil_div(payload_per_row, TRANSACTION_BYTES)
                        * TRANSACTION_BYTES) as u64,
                });
            }
            let mut issues = Vec::new();
            for m_start in (0..p.m).step_by(m_core) {
                for tile in &tiles {
                    issues.push(Issue {
                        m_start,
                        m_valid: (p.m - m_start).min(m_core),
                        n_start: tile.n_start,
                        n_valid: tile.n_valid,
                        k_start,
                        k_valid,
                    });
                }
            }
            groups.push(Group {
                id: groups.len(),
                n_start: p.n_start + n_offset,
                n_valid,
                k_start,
                k_valid,
                tiles,
                issues,
            });
        }
    }
    Ok(groups)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectionStats {
    pub groups: u64,
    pub weight_tiles: u64,
    pub issues: u64,
    pub weight_logical_bytes: u64,
    pub weight_native_bytes: u64,
    /// W reads into the compute operand path, repeated for successive M blocks.
    pub weight_gather_bytes: u64,
    /// X traffic into its SRAM slots for this explicit group traversal.
    pub x_staged_bytes: u64,
    pub x_loads: u64,
    /// Compulsory X payload read once by this owner, ignoring finite-slot reuse.
    /// It is a traffic lower bound, not an achieved flow or latency estimate.
    pub x_minimum_bytes: u64,
    /// X consumed by issues; this can exceed slot fills because of N reuse.
    pub x_operand_read_bytes: u64,
    pub useful_macs: u64,
    pub issued_macs: u64,
    pub tail_macs: u64,
    pub acc_read_bytes: u64,
    pub acc_write_bytes: u64,
    pub acc_rmw_bytes: u64,
    pub output_data_bytes: u64,
    /// Physical result row has four FP32 lanes even for a partial N tail.
    pub accumulator_data_reserved_bytes: u64,
    pub metadata_bytes: u64,
    pub accumulator_reserved_bytes: u64,
}

/// Summarizes a plan returned by `build_groups` for the same projection/core.
/// First K also uses an RMW, matching the current accumulator port convention.
/// Counts include per-output metadata, but not independent descriptor/control
/// state, which the global controller must account for separately.
pub fn summarize(p: Projection, m_core: usize, groups: &[Group]) -> ProjectionStats {
    let mut s = ProjectionStats::default();
    if p.m == 0 || p.n == 0 {
        return s;
    }
    s.groups = groups.len() as u64;
    s.x_minimum_bytes = p.m as u64 * p.k as u64 * BF16_BYTES as u64;
    s.output_data_bytes = p.m as u64 * p.n as u64 * FP32_BYTES as u64;
    let output_rows = p.m as u64 * ceil_div(p.n, N_TILE) as u64;
    s.accumulator_data_reserved_bytes = output_rows * OUTPUT_ROW_DATA_BYTES as u64;
    s.metadata_bytes = output_rows * OUTPUT_ROW_METADATA_BYTES as u64;
    s.accumulator_reserved_bytes = s.accumulator_data_reserved_bytes + s.metadata_bytes;
    for group in groups {
        s.weight_tiles += group.tiles.len() as u64;
        for tile in &group.tiles {
            s.weight_logical_bytes += tile.logical_bytes;
            s.weight_native_bytes += tile.native_bytes;
        }
        // Group traversal stages each real X row exactly once for this K
        // segment, regardless of the number of N bands served within the group.
        s.x_staged_bytes += p.m as u64 * group.k_valid as u64 * BF16_BYTES as u64;
        s.x_loads += ceil_div(p.m, m_core) as u64;
        for issue in &group.issues {
            s.issues += 1;
            s.useful_macs += issue.useful_macs();
            s.issued_macs += issue.issued_macs(m_core);
            s.weight_gather_bytes +=
                issue.n_valid as u64 * issue.k_valid as u64 * BF16_BYTES as u64;
            s.x_operand_read_bytes +=
                issue.m_valid as u64 * issue.k_valid as u64 * BF16_BYTES as u64;
            let result_bytes = issue.m_valid as u64 * issue.n_valid as u64 * FP32_BYTES as u64;
            s.acc_read_bytes += result_bytes;
            s.acc_write_bytes += result_bytes;
        }
    }
    s.tail_macs = s.issued_macs - s.useful_macs;
    s.acc_rmw_bytes = s.acc_read_bytes + s.acc_write_bytes;
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn ffn_shapes_preserve_hidden_and_intermediate_axes() {
        let p = ffn_projections(8, 2048, 1408);
        assert_eq!(
            p[0],
            (
                ProjectionKind::Gate,
                Projection {
                    m: 8,
                    n: 1408,
                    k: 2048,
                    n_start: 0
                }
            )
        );
        assert_eq!(p[1].0, ProjectionKind::Up);
        assert_eq!(p[1].1, p[0].1);
        assert_eq!(
            p[2],
            (
                ProjectionKind::Down,
                Projection {
                    m: 8,
                    n: 2048,
                    k: 1408,
                    n_start: 0
                }
            )
        );
    }

    #[test]
    fn group_traversal_reuses_x_and_preserves_every_outputs_k_order() {
        let p = Projection {
            m: 5,
            n: 10,
            k: 1025,
            n_start: 8,
        };
        let groups = build_groups(p, 4, 2).unwrap();
        let descriptors: Vec<_> = groups
            .iter()
            .map(|g| (g.n_start, g.n_valid, g.k_start, g.k_valid))
            .collect();
        assert_eq!(
            descriptors,
            vec![
                (8, 8, 0, 512),
                (8, 8, 512, 512),
                (8, 8, 1024, 1),
                (16, 2, 0, 512),
                (16, 2, 512, 512),
                (16, 2, 1024, 1)
            ]
        );
        let first: Vec<_> = groups[0]
            .issues
            .iter()
            .map(|i| (i.m_start, i.m_valid, i.n_start))
            .collect();
        assert_eq!(first, vec![(0, 4, 8), (0, 4, 12), (4, 1, 8), (4, 1, 12)]);
        let mut order = BTreeMap::<(usize, usize), Vec<usize>>::new();
        for group in &groups {
            assert!(group.tiles.len() <= 2);
            for i in &group.issues {
                for m in i.m_start..i.m_start + i.m_valid {
                    for n in i.n_start..i.n_start + i.n_valid {
                        order.entry((m, n)).or_default().push(i.k_start);
                    }
                }
            }
        }
        assert_eq!(order.len(), p.m * p.n);
        assert!(order.values().all(|v| v == &[0, 512, 1024]));
    }

    #[test]
    fn native_weight_padding_is_per_row_and_each_tile_loads_once() {
        let p = Projection {
            m: 5,
            n: 10,
            k: 1025,
            n_start: 0,
        };
        for m_core in [2, 3, 4, 6] {
            for g in [1, 2, 4] {
                let s = summarize(p, m_core, &build_groups(p, m_core, g).unwrap());
                assert_eq!(s.weight_logical_bytes, 2 * 10 * 1025);
                assert_eq!(s.weight_native_bytes, 10 * 2080); // 2048 + one 32 B tail / row
                assert_eq!(s.weight_tiles, 3 * 3);
                assert_eq!(s.useful_macs, 5 * 10 * 1025);
                assert_eq!(
                    s.weight_gather_bytes,
                    s.weight_logical_bytes * ceil_div(5, m_core) as u64
                );
            }
        }
    }

    #[test]
    fn grouping_changes_x_fills_but_never_weight_bytes_or_math() {
        let p = Projection {
            m: 8,
            n: 12,
            k: 1024,
            n_start: 0,
        };
        let mut all = Vec::new();
        for g in [1, 2, 4] {
            all.push(summarize(p, 4, &build_groups(p, 4, g).unwrap()));
        }
        assert_eq!(
            all.iter().map(|s| s.x_staged_bytes).collect::<Vec<_>>(),
            vec![49152, 32768, 16384]
        );
        assert!(all.iter().all(|s| s.x_minimum_bytes == 16384));
        assert!(all.iter().all(|s| s.weight_logical_bytes == 24576));
        assert!(
            all.iter()
                .all(|s| s.useful_macs == 98304 && s.issued_macs == 98304)
        );
        assert!(all.iter().all(|s| s.acc_rmw_bytes == 8 * 12 * 2 * 8));
        assert!(
            all.iter()
                .all(|s| s.output_data_bytes == 384 && s.accumulator_reserved_bytes == 768)
        );
    }

    #[test]
    fn counts_all_three_physical_tail_dimensions_without_changing_useful_math() {
        let p = Projection {
            m: 5,
            n: 5,
            k: 513,
            n_start: 0,
        };
        let s = summarize(p, 4, &build_groups(p, 4, 2).unwrap());
        assert_eq!(s.issues, 2 * 2 * 2);
        assert_eq!(s.useful_macs, 5 * 5 * 513);
        assert_eq!(s.issued_macs, 8 * 4 * 4 * 512);
        assert_eq!(s.tail_macs, s.issued_macs - s.useful_macs);
        assert_eq!(s.output_data_bytes, 100);
        assert_eq!(s.accumulator_data_reserved_bytes, 160);
        assert_eq!(s.metadata_bytes, 160);
        assert_eq!(s.acc_rmw_bytes, 5 * 5 * 2 * 8);
    }

    #[test]
    fn scalar_replay_matches_reference_for_m_n_and_k_tails() {
        // Integer-valued operands keep FP32 operations exact. This checks the
        // complete index coverage/order of the plan, not SRAM payload timing.
        let p = Projection {
            m: 5,
            n: 7,
            k: 513,
            n_start: 4,
        };
        let w_rows = p.n_start + p.n;
        let x: Vec<f32> = (0..p.m * p.k).map(|v| (v % 7) as f32 - 3.0).collect();
        let w: Vec<f32> = (0..w_rows * p.k).map(|v| (v % 5) as f32 - 2.0).collect();
        let mut reference = vec![0.0_f32; p.m * p.n];
        for m in 0..p.m {
            for n in 0..p.n {
                for k in 0..p.k {
                    reference[m * p.n + n] += x[m * p.k + k] * w[(p.n_start + n) * p.k + k];
                }
            }
        }
        for core in [2, 3, 4, 6] {
            for g in [1, 2, 4] {
                let mut out = vec![0.0_f32; p.m * p.n];
                for group in build_groups(p, core, g).unwrap() {
                    for i in group.issues {
                        for m in i.m_start..i.m_start + i.m_valid {
                            for n in i.n_start..i.n_start + i.n_valid {
                                for k in i.k_start..i.k_start + i.k_valid {
                                    out[m * p.n + n - p.n_start] += x[m * p.k + k] * w[n * p.k + k];
                                }
                            }
                        }
                    }
                }
                assert_eq!(out, reference, "core={core}, G={g}");
            }
        }
    }

    #[test]
    fn rejects_invalid_or_overflowed_geometry_and_skips_empty_experts() {
        let p = Projection {
            m: 1,
            n: 4,
            k: 512,
            n_start: 0,
        };
        assert_eq!(build_groups(p, 0, 1), Err(PlanError::ZeroCoreRows));
        assert_eq!(build_groups(p, 4, 0), Err(PlanError::ZeroGroupWidth));
        assert_eq!(
            build_groups(Projection { k: 0, ..p }, 4, 1),
            Err(PlanError::ZeroK)
        );
        assert_eq!(
            build_groups(
                Projection {
                    n_start: usize::MAX,
                    ..p
                },
                4,
                1
            ),
            Err(PlanError::DimensionOverflow)
        );
        assert_eq!(
            build_groups(Projection { m: usize::MAX, ..p }, 4, 1),
            Err(PlanError::DimensionOverflow)
        );
        let empty = Projection { m: 0, ..p };
        assert!(build_groups(empty, 4, 2).unwrap().is_empty());
        assert_eq!(summarize(empty, 4, &[]), ProjectionStats::default());
    }
}
