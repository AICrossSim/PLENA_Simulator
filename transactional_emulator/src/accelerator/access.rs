//! Per-opcode dependency descriptors for the pipelined timing scoreboard.
//!
//! [`op_access`] maps one decoded opcode (plus the register values it will
//! consume at issue) to the full set of architectural resources it reads and
//! writes: general-purpose / floating-point / HBM-address registers, sticky
//! config registers, matrix-accumulator state, and matrix/vector/scalar SRAM
//! ranges. The analytic scoreboard (`accelerator::scoreboard`) uses these sets
//! to decide when an instruction may issue.
//!
//! # Invariant: this mirrors the execution arms in `do_ops`
//!
//! Like `classify_timing_access` (which covers only the prefetch-overlap
//! subset and stays untouched while the serial overlay still exists), the
//! extents here are hand-derived from the execution arms and the machine
//! methods they call. Any change to an arm's register or SRAM addressing must
//! be reflected here. The guards are the same: the match is exhaustive with no
//! `_` arm, zero-cost no-op arms are mirrored explicitly, and the function is
//! free of `&self` so it is unit-testable. Two further safety nets exist at
//! runtime: DMA destinations are genuinely `Cell::Pending` (an unclassified
//! read still blocks functionally), and dispatch warns whenever an arm
//! advanced the virtual clock past its modeled issue instant.
//!
//! Timing semantics the sets imply (see `scoreboard`):
//! - reads → RAW: issue waits for the writers' `ready_at`.
//! - writes → WAW: `ready_at` is max-merged at commit; no issue stall.
//! - WAR needs no stall: in-order issue means a later writer cannot be
//!   modeled as landing before an earlier reader's issue-time operand fetch.

use crate::op;
use crate::runtime_config::{
    BLEN, BROADCAST_AMOUNT, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT, STORE_V_AMOUNT, VLEN,
};

/// Which address space an SRAM range lives in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SramSpace {
    Matrix,
    Vector,
    ScalarFp,
    ScalarInt,
}

impl SramSpace {
    pub(crate) const COUNT: usize = 4;

    pub(crate) fn index(self) -> usize {
        match self {
            SramSpace::Matrix => 0,
            SramSpace::Vector => 1,
            SramSpace::ScalarFp => 2,
            SramSpace::ScalarInt => 3,
        }
    }
}

/// A half-open element-address range `[start, start + len)` in one SRAM space.
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

    pub(crate) fn overlaps(self, other: Self) -> bool {
        if self.space != other.space || self.len == 0 || other.len == 0 {
            return false;
        }
        let self_end = self.start.saturating_add(self.len);
        let other_end = other.start.saturating_add(other.len);
        self.start < other_end && other.start < self_end
    }
}

/// Sticky config registers written by `C_SET_*_REG` opcodes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Cfg {
    Scale,
    Stride,
    VMask,
    TopkPolicy,
    LStream,
}

impl Cfg {
    pub(crate) const COUNT: usize = 5;

    pub(crate) fn index(self) -> usize {
        match self {
            Cfg::Scale => 0,
            Cfg::Stride => 1,
            Cfg::VMask => 2,
            Cfg::TopkPolicy => 3,
            Cfg::LStream => 4,
        }
    }
}

/// The four matrix-machine accumulators. Accumulate ops read+write their
/// accumulator; the matching `*_WO` flushes it (also read+write). With a
/// single matrix core the unit `busy_until` already serializes these; the
/// tokens make the dependency explicit and future-proof multi-core setups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AccumKind {
    /// `m_accum` (M_MM / M_TMM / M_MM_WO).
    M,
    /// `hm_accum` (M_BMM / M_BTMM / M_BMM_WO).
    Hm,
    /// `hv_accum` (M_BMV / M_BTMV / M_BMV_WO).
    Hv,
    /// `v_accum` (M_MV / M_TMV / M_MV_WO).
    V,
}

impl AccumKind {
    pub(crate) const COUNT: usize = 4;

    pub(crate) fn index(self) -> usize {
        match self {
            AccumKind::M => 0,
            AccumKind::Hm => 1,
            AccumKind::Hv => 2,
            AccumKind::V => 3,
        }
    }
}

/// One architectural resource an opcode reads or writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Resource {
    Gp(u8),
    Fp(u8),
    Hbm(u8),
    Cfg(Cfg),
    Accum(AccumKind),
    Sram(SramRange),
}

/// The functional unit an opcode occupies while executing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Unit {
    Matrix,
    Vector,
    Scalar,
    Dma,
}

impl Unit {
    pub(crate) const COUNT: usize = 4;

    pub(crate) fn index(self) -> usize {
        match self {
            Unit::Matrix => 0,
            Unit::Vector => 1,
            Unit::Scalar => 2,
            Unit::Dma => 3,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Unit::Matrix => "matrix",
            Unit::Vector => "vector",
            Unit::Scalar => "scalar",
            Unit::Dma => "dma",
        }
    }
}

/// Everything the scoreboard needs to know about one dynamic instruction.
#[derive(Clone, Debug)]
pub(crate) struct OpAccess {
    pub(crate) unit: Unit,
    /// `C_BREAK`: issue only after everything previously issued has finished.
    pub(crate) barrier: bool,
    pub(crate) reads: Vec<Resource>,
    pub(crate) writes: Vec<Resource>,
}

impl OpAccess {
    fn new(unit: Unit, reads: Vec<Resource>, writes: Vec<Resource>) -> Self {
        Self {
            unit,
            barrier: false,
            reads,
            writes,
        }
    }

    /// A front-end-only op: occupies no unit state beyond the issue slot.
    fn none(unit: Unit) -> Self {
        Self::new(unit, vec![], vec![])
    }
}

/// Build the access descriptor for `op`, sampling `gp` register values (they
/// form SRAM addresses) and the sticky topk policy exactly as the execution
/// arm will at issue.
pub(crate) fn op_access(
    op: &op::Opcode,
    gp: &dyn Fn(u8) -> u32,
    topk_policy: &dyn Fn() -> Option<(usize, usize)>,
) -> OpAccess {
    use Resource::{Accum, Fp, Gp, Hbm, Sram};

    let matrix_tile = *MLEN * *MLEN;
    let vector_tile = *VLEN;

    let matrix = |start: u32, len: u32| Sram(SramRange::new(SramSpace::Matrix, start, len));
    let vector = |start: u32, len: u32| Sram(SramRange::new(SramSpace::Vector, start, len));
    let scalar_fp = |start: u32, len: u32| Sram(SramRange::new(SramSpace::ScalarFp, start, len));
    let scalar_int = |start: u32, len: u32| Sram(SramRange::new(SramSpace::ScalarInt, start, len));

    // Matrix reads resolve whole tiles: align the element address down to its
    // tile base so the range covers the exact cell the machine locks.
    let matrix_tile_at = |addr: u32| {
        Sram(SramRange::new(
            SramSpace::Matrix,
            addr - addr % matrix_tile,
            matrix_tile,
        ))
    };
    // `MatrixMachine` write-outs align the vram address down to a whole row
    // (`multiple_and_offset(v_addr, mlen)`) before touching vram.
    let row_base = |addr: u32| addr - (addr % *MLEN);
    // Masked V_* ops consult the sticky v_mask register; rmask == 0 uses the
    // constant all-heads mask and reads no config state.
    let mask_read = |rmask: u8, reads: &mut Vec<Resource>| {
        if rmask != 0 {
            reads.push(Resource::Cfg(Cfg::VMask));
        }
    };

    match *op {
        op::Opcode::Invalid => OpAccess::none(Unit::Scalar),

        // === Matrix accumulate ops ===
        op::Opcode::M_MM { rs1, rs2 } | op::Opcode::M_TMM { rs1, rs2 } => OpAccess::new(
            Unit::Matrix,
            vec![
                Gp(rs1),
                Gp(rs2),
                matrix_tile_at(gp(rs1)),
                vector(gp(rs2), *MLEN * *BLEN),
                Accum(AccumKind::M),
            ],
            vec![Accum(AccumKind::M)],
        ),
        op::Opcode::M_BMM { rs1, rs2 } | op::Opcode::M_BTMM { rs1, rs2 } => OpAccess::new(
            Unit::Matrix,
            vec![
                Gp(rs1),
                Gp(rs2),
                matrix_tile_at(gp(rs1)),
                vector(gp(rs2), matrix_tile),
                Accum(AccumKind::Hm),
            ],
            vec![Accum(AccumKind::Hm)],
        ),
        op::Opcode::M_MV { rs1, rs2 } | op::Opcode::M_TMV { rs1, rs2 } => OpAccess::new(
            Unit::Matrix,
            vec![
                Gp(rs1),
                Gp(rs2),
                matrix_tile_at(gp(rs1)),
                vector(gp(rs2), vector_tile),
                Accum(AccumKind::V),
            ],
            vec![Accum(AccumKind::V)],
        ),
        op::Opcode::M_BMV { rs1, rs2, rd } | op::Opcode::M_BTMV { rs1, rs2, rd } => OpAccess::new(
            Unit::Matrix,
            vec![
                Gp(rs1),
                Gp(rs2),
                Gp(rd),
                matrix_tile_at(gp(rs1).wrapping_add(gp(rd))),
                vector(gp(rs2), vector_tile),
                Accum(AccumKind::Hv),
            ],
            vec![Accum(AccumKind::Hv)],
        ),

        // === Matrix write-outs ===
        // `mm_wo` is a read-modify-write: for each of `blen` rows it reads
        // `vec_base + i * mlen * stride_len`, splices the accumulator in, and
        // writes the row back.
        op::Opcode::M_MM_WO { rd, rstride, imm } => {
            let stride_len = if rstride == 0 { 1 } else { gp(rstride) };
            let base = row_base(gp(rd).wrapping_add(imm));
            let span = (*BLEN)
                .saturating_sub(1)
                .saturating_mul(*MLEN)
                .saturating_mul(stride_len)
                .saturating_add(vector_tile);
            let mut reads = vec![Gp(rd), Accum(AccumKind::M), vector(base, span)];
            if rstride != 0 {
                reads.push(Gp(rstride));
            }
            OpAccess::new(
                Unit::Matrix,
                reads,
                vec![Accum(AccumKind::M), vector(base, span)],
            )
        }
        // `mv_wo` reads exactly the one destination row before splicing.
        op::Opcode::M_MV_WO { rd, imm } => {
            let base = row_base(gp(rd).wrapping_add(imm));
            OpAccess::new(
                Unit::Matrix,
                vec![Gp(rd), Accum(AccumKind::V), vector(base, vector_tile)],
                vec![Accum(AccumKind::V), vector(base, vector_tile)],
            )
        }
        // `bmm_wo` overwrites `broadcast_amount * mlen` contiguous rows.
        op::Opcode::M_BMM_WO { rd, imm } => {
            let base = row_base(gp(rd).wrapping_add(imm));
            let span = (*BROADCAST_AMOUNT).saturating_mul(matrix_tile);
            OpAccess::new(
                Unit::Matrix,
                vec![Gp(rd), Accum(AccumKind::Hm)],
                vec![Accum(AccumKind::Hm), vector(base, span)],
            )
        }
        // `bmv_wo` overwrites `broadcast_amount` rows spaced `mlen` apart.
        op::Opcode::M_BMV_WO { rd, imm } => {
            let base = row_base(gp(rd).wrapping_add(imm));
            let span = (*BROADCAST_AMOUNT)
                .saturating_sub(1)
                .saturating_mul(*MLEN)
                .saturating_add(vector_tile);
            OpAccess::new(
                Unit::Matrix,
                vec![Gp(rd), Accum(AccumKind::Hv)],
                vec![Accum(AccumKind::Hv), vector(base, span)],
            )
        }

        // === Vector ops: two vram sources ===
        op::Opcode::V_ADD_VV {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        }
        | op::Opcode::V_SUB_VV {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        }
        | op::Opcode::V_MUL_VV {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        } => {
            let mut reads = vec![
                Gp(rd),
                Gp(rs1),
                Gp(rs2),
                vector(gp(rs1), vector_tile),
                vector(gp(rs2), vector_tile),
            ];
            mask_read(rmask, &mut reads);
            OpAccess::new(Unit::Vector, reads, vec![vector(gp(rd), vector_tile)])
        }

        // === Vector ops: one vram source + fp scalar ===
        op::Opcode::V_ADD_VF {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        }
        | op::Opcode::V_SUB_VF {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        }
        | op::Opcode::V_MUL_VF {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        }
        | op::Opcode::V_MAX_VF {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        }
        | op::Opcode::V_MIN_VF {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        } => {
            let mut reads = vec![Gp(rd), Gp(rs1), Fp(rs2), vector(gp(rs1), vector_tile)];
            mask_read(rmask, &mut reads);
            OpAccess::new(Unit::Vector, reads, vec![vector(gp(rd), vector_tile)])
        }

        // V_FMA_VF is the VF family plus one thing: it reads its **destination**
        // as well as its source, because `V[rd] += V[rs1] * fp[rs2]`. Grouping it
        // with the arm above would under-report its vector-SRAM traffic by a row,
        // which is precisely the number the FMA conversion is judged on.
        op::Opcode::V_FMA_VF {
            rd,
            rs1,
            rs2,
            rmask,
            ..
        } => {
            let mut reads = vec![
                Gp(rd),
                Gp(rs1),
                Fp(rs2),
                vector(gp(rs1), vector_tile),
                vector(gp(rd), vector_tile),
            ];
            mask_read(rmask, &mut reads);
            OpAccess::new(Unit::Vector, reads, vec![vector(gp(rd), vector_tile)])
        }

        // === Vector ops: one vram source, vram dest ===
        op::Opcode::V_EXP_V { rd, rs1, rmask, .. }
        | op::Opcode::V_RECI_V { rd, rs1, rmask, .. }
        | op::Opcode::V_SOFTPLUS_V { rd, rs1, rmask, .. } => {
            let mut reads = vec![Gp(rd), Gp(rs1), vector(gp(rs1), vector_tile)];
            mask_read(rmask, &mut reads);
            OpAccess::new(Unit::Vector, reads, vec![vector(gp(rd), vector_tile)])
        }
        op::Opcode::V_SHFT_V { rd, rs1, rs2 } => OpAccess::new(
            Unit::Vector,
            vec![Gp(rd), Gp(rs1), Gp(rs2), vector(gp(rs1), vector_tile)],
            vec![vector(gp(rd), vector_tile)],
        ),

        // Writing to fp0 is discarded; the execution arm returns without
        // touching vram or charging cycles. Mirror as a front-end no-op.
        op::Opcode::V_RED_SUM { rd: 0, .. } | op::Opcode::V_RED_MAX { rd: 0, .. } => {
            OpAccess::none(Unit::Scalar)
        }

        // Reductions read the fp destination as their initial value and write
        // the result back to it.
        op::Opcode::V_RED_SUM { rd, rs1, rmask, .. }
        | op::Opcode::V_RED_MAX { rd, rs1, rmask, .. } => {
            let mut reads = vec![Gp(rs1), Fp(rd), vector(gp(rs1), vector_tile)];
            mask_read(rmask, &mut reads);
            OpAccess::new(Unit::Vector, reads, vec![Fp(rd)])
        }

        // `topk_softmax` walks the logits in VLEN-sized chunks, then dispatch
        // scatters `topk` indices/weights into scalar SRAM.
        op::Opcode::V_TOPK {
            rd,
            rs1,
            rs2,
            rmask,
        } => {
            let (expert_count, topk): (u32, u32) = match rmask {
                0 => (32, 4),
                1 => (128, 8),
                // rmask=15 takes its shape from C_SET_TOPK_REG; unset traps in
                // the execution arm, so fall back conservatively here.
                15 => topk_policy().map_or((128, 8), |(experts, k)| (experts as u32, k as u32)),
                // Unsupported policies panic in the execution arm.
                _ => (128, 8),
            };
            let rows = expert_count.div_ceil(vector_tile.max(1));
            let mut reads = vec![
                Gp(rd),
                Gp(rs1),
                Gp(rs2),
                vector(gp(rs1), rows * vector_tile),
            ];
            if rmask == 15 {
                reads.push(Resource::Cfg(Cfg::TopkPolicy));
            }
            OpAccess::new(
                Unit::Vector,
                reads,
                vec![scalar_int(gp(rs2), topk), scalar_fp(gp(rd), topk)],
            )
        }

        // Writing to fp0 is discarded; the execution arm returns immediately.
        op::Opcode::S_ADD_FP { rd: 0, .. }
        | op::Opcode::S_SUB_FP { rd: 0, .. }
        | op::Opcode::S_MAX_FP { rd: 0, .. }
        | op::Opcode::S_MUL_FP { rd: 0, .. }
        | op::Opcode::S_EXP_FP { rd: 0, .. }
        | op::Opcode::S_RECI_FP { rd: 0, .. }
        | op::Opcode::S_SQRT_FP { rd: 0, .. } => OpAccess::none(Unit::Scalar),

        op::Opcode::S_ADD_FP { rd, rs1, rs2 }
        | op::Opcode::S_SUB_FP { rd, rs1, rs2 }
        | op::Opcode::S_MAX_FP { rd, rs1, rs2 }
        | op::Opcode::S_MUL_FP { rd, rs1, rs2 } => {
            OpAccess::new(Unit::Scalar, vec![Fp(rs1), Fp(rs2)], vec![Fp(rd)])
        }
        op::Opcode::S_EXP_FP { rd, rs1 }
        | op::Opcode::S_RECI_FP { rd, rs1 }
        | op::Opcode::S_SQRT_FP { rd, rs1 } => {
            OpAccess::new(Unit::Scalar, vec![Fp(rs1)], vec![Fp(rd)])
        }
        // Note: unlike the arithmetic fp ops above, S_LD_FP rd=0 really does
        // write fp0 in dispatch (no no-op arm exists for it) — mirrored as-is.
        op::Opcode::S_LD_FP { rd, rs1, imm } => OpAccess::new(
            Unit::Scalar,
            vec![Gp(rs1), scalar_fp(gp(rs1).wrapping_add(imm), 1)],
            vec![Fp(rd)],
        ),
        op::Opcode::S_ST_FP { rd, rs1, imm } => OpAccess::new(
            Unit::Scalar,
            vec![Fp(rd), Gp(rs1)],
            vec![scalar_fp(gp(rs1).wrapping_add(imm), 1)],
        ),
        op::Opcode::S_MAP_V_FP { rd, rs1, imm } => OpAccess::new(
            Unit::Scalar,
            vec![
                Gp(rd),
                Gp(rs1),
                scalar_fp(gp(rs1).wrapping_add(imm), vector_tile),
            ],
            vec![vector(gp(rd), vector_tile)],
        ),

        // The mirror of S_MAP_V_FP above, with every role inverted: `rs1` is the
        // VRAM row it reads, `rd` the FP_MEM base it writes. It is billed to the
        // Vector unit rather than Scalar because it holds the vector SRAM read
        // port for a whole row.
        op::Opcode::S_MAP_FP_V { rd, rs1, imm } => OpAccess::new(
            Unit::Vector,
            vec![Gp(rd), Gp(rs1), vector(gp(rs1), vector_tile)],
            vec![scalar_fp(gp(rd).wrapping_add(imm), vector_tile)],
        ),

        op::Opcode::S_ADD_INT { rd, rs1, rs2 }
        | op::Opcode::S_SUB_INT { rd, rs1, rs2 }
        | op::Opcode::S_MUL_INT { rd, rs1, rs2 } => {
            OpAccess::new(Unit::Scalar, vec![Gp(rs1), Gp(rs2)], vec![Gp(rd)])
        }
        op::Opcode::S_ADDI_INT { rd, rs1, .. } => {
            OpAccess::new(Unit::Scalar, vec![Gp(rs1)], vec![Gp(rd)])
        }
        op::Opcode::S_LUI_INT { rd, .. } => OpAccess::new(Unit::Scalar, vec![], vec![Gp(rd)]),
        op::Opcode::S_LD_INT { rd, rs1, imm } => OpAccess::new(
            Unit::Scalar,
            vec![Gp(rs1), scalar_int(gp(rs1).wrapping_add(imm), 1)],
            vec![Gp(rd)],
        ),
        op::Opcode::S_ST_INT { rd, rs1, imm } => OpAccess::new(
            Unit::Scalar,
            vec![Gp(rd), Gp(rs1)],
            vec![scalar_int(gp(rs1).wrapping_add(imm), 1)],
        ),

        // === HBM DMA ===
        op::Opcode::H_PREFETCH_M { rd, rs1, rs2, .. } => OpAccess::new(
            Unit::Dma,
            vec![
                Gp(rd),
                Gp(rs1),
                Hbm(rs2),
                Resource::Cfg(Cfg::Scale),
                Resource::Cfg(Cfg::Stride),
            ],
            vec![matrix(gp(rd), *MLEN * *PREFETCH_M_AMOUNT)],
        ),
        op::Opcode::H_PREFETCH_V { rd, rs1, rs2, .. } => OpAccess::new(
            Unit::Dma,
            vec![
                Gp(rd),
                Gp(rs1),
                Hbm(rs2),
                Resource::Cfg(Cfg::Scale),
                Resource::Cfg(Cfg::Stride),
            ],
            vec![vector(gp(rd), *VLEN * *PREFETCH_V_AMOUNT)],
        ),
        // A store reads the vram region it drains. Its HBM-side write is not
        // tracked as a resource; dispatch conservatively drains all pending
        // prefetches before an H_STORE_V instead (HBM WAR/RAW).
        op::Opcode::H_STORE_V { rd, rs1, rs2, .. } => OpAccess::new(
            Unit::Dma,
            vec![
                Gp(rd),
                Gp(rs1),
                Hbm(rs2),
                Resource::Cfg(Cfg::Scale),
                Resource::Cfg(Cfg::Stride),
                vector(gp(rd), *VLEN * *STORE_V_AMOUNT),
            ],
            vec![],
        ),

        // === Control ===
        op::Opcode::C_SET_ADDR_REG { rd, rs1, rs2 } => {
            OpAccess::new(Unit::Scalar, vec![Gp(rs1), Gp(rs2)], vec![Hbm(rd)])
        }
        op::Opcode::C_SET_SCALE_REG { rd } => {
            OpAccess::new(Unit::Scalar, vec![Gp(rd)], vec![Resource::Cfg(Cfg::Scale)])
        }
        op::Opcode::C_SET_STRIDE_REG { rd } => {
            OpAccess::new(Unit::Scalar, vec![Gp(rd)], vec![Resource::Cfg(Cfg::Stride)])
        }
        op::Opcode::C_SET_V_MASK_REG { rd } => {
            OpAccess::new(Unit::Scalar, vec![Gp(rd)], vec![Resource::Cfg(Cfg::VMask)])
        }
        op::Opcode::C_SET_TOPK_REG { rd } => OpAccess::new(
            Unit::Scalar,
            vec![Gp(rd)],
            vec![Resource::Cfg(Cfg::TopkPolicy)],
        ),
        op::Opcode::L_CFG { value, .. } => OpAccess::new(
            Unit::Scalar,
            vec![Gp(value)],
            vec![Resource::Cfg(Cfg::LStream)],
        ),
        op::Opcode::C_LOOP_START { rd, .. } => OpAccess::new(Unit::Scalar, vec![], vec![Gp(rd)]),
        op::Opcode::C_LOOP_END { rd } => OpAccess::new(Unit::Scalar, vec![Gp(rd)], vec![Gp(rd)]),
        // C_BREAK writes the innermost loop's counter register, which is only
        // known from loop state; modeling it as a full barrier dominates any
        // per-register dependency it could carry.
        op::Opcode::C_BREAK => {
            let mut access = OpAccess::none(Unit::Scalar);
            access.barrier = true;
            access
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::VectorPrecision;

    /// Register file stand-in: `gp(n)` returns `n as u32 * 1024`.
    fn gp_stub(reg: u8) -> u32 {
        reg as u32 * 1024
    }

    fn access(op: op::Opcode) -> OpAccess {
        op_access(&op, &gp_stub, &|| None)
    }

    fn sram_ranges(resources: &[Resource]) -> Vec<(SramSpace, u32, u32)> {
        resources
            .iter()
            .filter_map(|r| match r {
                Resource::Sram(range) => Some((range.space, range.start, range.len)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn matrix_multiply_reads_regs_tile_batch_and_accumulator() {
        let a = access(op::Opcode::M_MM { rs1: 1, rs2: 2 });
        assert_eq!(a.unit, Unit::Matrix);
        assert!(a.reads.contains(&Resource::Gp(1)));
        assert!(a.reads.contains(&Resource::Gp(2)));
        assert!(a.reads.contains(&Resource::Accum(AccumKind::M)));
        assert_eq!(a.writes, vec![Resource::Accum(AccumKind::M)]);
        let tile = *MLEN * *MLEN;
        let base = gp_stub(1) - gp_stub(1) % tile;
        assert_eq!(
            sram_ranges(&a.reads),
            vec![
                (SramSpace::Matrix, base, tile),
                (SramSpace::Vector, gp_stub(2), *MLEN * *BLEN),
            ]
        );
    }

    #[test]
    fn matrix_write_out_reads_and_writes_its_destination_rows() {
        let a = access(op::Opcode::M_MM_WO {
            rd: 2,
            rstride: 0,
            imm: 0,
        });
        let expected_span = (*BLEN - 1) * *MLEN + *VLEN;
        assert_eq!(
            sram_ranges(&a.reads),
            vec![(SramSpace::Vector, gp_stub(2), expected_span)]
        );
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(SramSpace::Vector, gp_stub(2), expected_span)]
        );
        assert!(a.reads.contains(&Resource::Accum(AccumKind::M)));
        assert!(a.writes.contains(&Resource::Accum(AccumKind::M)));
        // rstride == 0 means stride 1 without reading a register.
        assert!(!a.reads.contains(&Resource::Gp(0)));
    }

    #[test]
    fn strided_write_out_reads_its_stride_register() {
        let a = access(op::Opcode::M_MM_WO {
            rd: 2,
            rstride: 3,
            imm: 0,
        });
        assert!(a.reads.contains(&Resource::Gp(3)));
        let expected_span = (*BLEN - 1) * *MLEN * gp_stub(3) + *VLEN;
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(SramSpace::Vector, gp_stub(2), expected_span)]
        );
    }

    #[test]
    fn batched_write_outs_write_their_rows_without_reading_them() {
        let a = access(op::Opcode::M_BMM_WO { rd: 2, imm: 0 });
        assert!(sram_ranges(&a.reads).is_empty());
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(
                SramSpace::Vector,
                gp_stub(2),
                *BROADCAST_AMOUNT * *MLEN * *MLEN
            )]
        );

        let a = access(op::Opcode::M_BMV_WO { rd: 2, imm: 0 });
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(
                SramSpace::Vector,
                gp_stub(2),
                (*BROADCAST_AMOUNT - 1) * *MLEN + *VLEN
            )]
        );
    }

    #[test]
    fn vector_binary_op_reads_sources_and_writes_destination() {
        let a = access(op::Opcode::V_ADD_VV {
            rd: 1,
            rs1: 2,
            rs2: 3,
            rmask: 0,
            lmask: 0,
        });
        assert_eq!(a.unit, Unit::Vector);
        assert_eq!(
            sram_ranges(&a.reads),
            vec![
                (SramSpace::Vector, gp_stub(2), *VLEN),
                (SramSpace::Vector, gp_stub(3), *VLEN),
            ]
        );
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(SramSpace::Vector, gp_stub(1), *VLEN)]
        );
        // rmask == 0 uses the constant all-heads mask.
        assert!(!a.reads.contains(&Resource::Cfg(Cfg::VMask)));

        let masked = access(op::Opcode::V_ADD_VV {
            rd: 1,
            rs1: 2,
            rs2: 3,
            rmask: 1,
            lmask: 0,
        });
        assert!(masked.reads.contains(&Resource::Cfg(Cfg::VMask)));
    }

    #[test]
    fn reductions_read_and_write_their_fp_register() {
        let a = access(op::Opcode::V_RED_SUM {
            rd: 3,
            rs1: 2,
            rmask: 0,
            lmask: 0,
        });
        // The reduction folds the fp register's current value in as its
        // initial accumulator, so fp(rd) is both read and written.
        assert!(a.reads.contains(&Resource::Fp(3)));
        assert_eq!(a.writes, vec![Resource::Fp(3)]);

        // fp0 destination is a no-op arm.
        let noop = access(op::Opcode::V_RED_SUM {
            rd: 0,
            rs1: 2,
            rmask: 0,
            lmask: 0,
        });
        assert!(noop.reads.is_empty() && noop.writes.is_empty());
    }

    #[test]
    fn topk_writes_scalar_srams_and_reads_every_logit_row() {
        let a = op_access(
            &op::Opcode::V_TOPK {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 15,
            },
            &gp_stub,
            &|| Some((256, 8)),
        );
        let rows = 256u32.div_ceil(*VLEN);
        assert_eq!(
            sram_ranges(&a.reads),
            vec![(SramSpace::Vector, gp_stub(2), rows * *VLEN)]
        );
        assert_eq!(
            sram_ranges(&a.writes),
            vec![
                (SramSpace::ScalarInt, gp_stub(3), 8),
                (SramSpace::ScalarFp, gp_stub(1), 8),
            ]
        );
        assert!(a.reads.contains(&Resource::Cfg(Cfg::TopkPolicy)));
    }

    #[test]
    fn prefetch_reads_its_address_registers_and_writes_the_sram_fill() {
        let a = access(op::Opcode::H_PREFETCH_V {
            rd: 2,
            rs1: 3,
            rs2: 4,
            rstride: 0,
            precision: VectorPrecision::Activation,
        });
        assert_eq!(a.unit, Unit::Dma);
        assert!(a.reads.contains(&Resource::Gp(2)));
        assert!(a.reads.contains(&Resource::Gp(3)));
        assert!(a.reads.contains(&Resource::Hbm(4)));
        assert!(a.reads.contains(&Resource::Cfg(Cfg::Scale)));
        assert!(a.reads.contains(&Resource::Cfg(Cfg::Stride)));
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(SramSpace::Vector, gp_stub(2), *VLEN * *PREFETCH_V_AMOUNT)]
        );
    }

    #[test]
    fn store_reads_the_region_it_drains() {
        let a = access(op::Opcode::H_STORE_V {
            rd: 3,
            rs1: 0,
            rs2: 0,
            rstride: 0,
            precision: VectorPrecision::Activation,
        });
        assert_eq!(a.unit, Unit::Dma);
        assert_eq!(
            sram_ranges(&a.reads),
            vec![(SramSpace::Vector, gp_stub(3), *VLEN * *STORE_V_AMOUNT)]
        );
        assert!(sram_ranges(&a.writes).is_empty());
    }

    #[test]
    fn scalar_ops_track_register_flow() {
        let a = access(op::Opcode::S_ADD_INT {
            rd: 1,
            rs1: 2,
            rs2: 3,
        });
        assert_eq!(a.reads, vec![Resource::Gp(2), Resource::Gp(3)]);
        assert_eq!(a.writes, vec![Resource::Gp(1)]);

        let a = access(op::Opcode::S_LD_FP {
            rd: 2,
            rs1: 3,
            imm: 8,
        });
        assert_eq!(
            sram_ranges(&a.reads),
            vec![(SramSpace::ScalarFp, gp_stub(3) + 8, 1)]
        );
        assert_eq!(a.writes, vec![Resource::Fp(2)]);

        // fp0 arithmetic destinations are no-ops...
        let noop = access(op::Opcode::S_ADD_FP {
            rd: 0,
            rs1: 1,
            rs2: 2,
        });
        assert!(noop.reads.is_empty() && noop.writes.is_empty());
    }

    #[test]
    fn map_v_fp_bridges_scalar_fp_sram_into_vram() {
        let a = access(op::Opcode::S_MAP_V_FP {
            rd: 1,
            rs1: 2,
            imm: 4,
        });
        assert_eq!(
            sram_ranges(&a.reads),
            vec![(SramSpace::ScalarFp, gp_stub(2) + 4, *VLEN)]
        );
        assert_eq!(
            sram_ranges(&a.writes),
            vec![(SramSpace::Vector, gp_stub(1), *VLEN)]
        );
    }

    #[test]
    fn loop_ops_carry_their_counter_register_and_break_is_a_barrier() {
        let start = access(op::Opcode::C_LOOP_START { rd: 5, imm: 3 });
        assert_eq!(start.writes, vec![Resource::Gp(5)]);

        let end = access(op::Opcode::C_LOOP_END { rd: 5 });
        assert_eq!(end.reads, vec![Resource::Gp(5)]);
        assert_eq!(end.writes, vec![Resource::Gp(5)]);
        assert!(!end.barrier);

        assert!(access(op::Opcode::C_BREAK).barrier);
    }

    #[test]
    fn config_setters_write_their_sticky_registers() {
        let a = access(op::Opcode::C_SET_SCALE_REG { rd: 1 });
        assert_eq!(a.reads, vec![Resource::Gp(1)]);
        assert_eq!(a.writes, vec![Resource::Cfg(Cfg::Scale)]);

        let a = access(op::Opcode::C_SET_ADDR_REG {
            rd: 1,
            rs1: 2,
            rs2: 3,
        });
        assert_eq!(a.writes, vec![Resource::Hbm(1)]);
    }

    #[test]
    fn range_overlap_is_per_space_and_half_open() {
        let a = SramRange::new(SramSpace::Vector, 0, 64);
        let b = SramRange::new(SramSpace::Vector, 63, 1);
        let c = SramRange::new(SramSpace::Vector, 64, 64);
        let d = SramRange::new(SramSpace::Matrix, 0, 64);
        assert!(a.overlaps(b));
        assert!(!a.overlaps(c));
        assert!(!a.overlaps(d));
        assert!(!a.overlaps(SramRange::new(SramSpace::Vector, 0, 0)));
    }
}
