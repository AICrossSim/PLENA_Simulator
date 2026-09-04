//! Opcode execution for [`Accelerator`].
//!
//! The public accelerator facade stays in `mod.rs`; this module owns the ISA
//! match and dispatch-only helpers.

use half::bf16;
use quantize::{MxDataType, QuantTensor, tensor_from_f32_slice, tensor_to_f32_vec};

use crate::runtime_config::PERIOD;
use crate::runtime_config::{
    HLEN, MATRIX_KV_TYPE, MATRIX_WEIGHT_TYPE, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT,
    SCALAR_FP_BASIC_CYCLES, SCALAR_FP_EXP_CYCLES, SCALAR_FP_RECI_CYCLES, SCALAR_FP_SQRT_CYCLES,
    SCALAR_INT_BASIC_CYCLES, STATE_TYPE, STORE_V_AMOUNT, VECTOR_ACTIVATION_TYPE, VECTOR_KV_TYPE,
    VLEN,
};
use crate::stage_profile::{ResourceKind, StageProfiler};
use crate::vector_machine::{ScalarOperand, VectorBinaryOp, VectorOperandViews};
use crate::{cycle, dma, op, timing};
use runtime::{Executor, Instant};
use sram::matrix::MatrixPacketService;

use super::Accelerator;
use super::access::{self, OpAccess};
use super::loop_state::LoopDecision;
use super::lstream::{ConfigField, StreamTarget};
use super::mview::MatrixViewDescriptor;
use super::scoreboard::{DmaKind, Scoreboard};

/// How `do_ops` charges time.
///
/// `Serial` is the historical model: every opcode sleeps for its full latency
/// on the dispatch task, so total cycles are the sum of all per-instruction
/// latencies. `Scoreboard` keeps functional execution serial but models
/// pipelined issue via the analytic scoreboard: dispatch sleeps only to each
/// op's modeled issue instant and the final clock is the scoreboard's
/// max-finish.
pub(crate) enum TimingDriver<'a> {
    Serial,
    Scoreboard { scoreboard: &'a mut Scoreboard },
}

struct MatrixViewBinaryArgs {
    operation: VectorBinaryOp,
    rd: u8,
    rs1: u8,
    rs2: u8,
    rmask: u8,
    encoded_view_mask: u8,
    mask: u32,
    pc: usize,
}

struct LTileExecArgs {
    destination_register: u8,
    source_register: u8,
    scale_register: u8,
    primitive: op::LTilePrimitive,
    source_axis: op::LTileAxis,
    scale_axis: op::LTileAxis,
}

impl Accelerator {
    /// Resolve the V_* opcode mask.
    ///
    /// When `rmask == 0`, the opcode operates on all HLEN heads of the VLEN
    /// vector (mask = all-ones over `*HLEN` bits). Otherwise the per-head mask
    /// stored in `reg_file.v_mask` is used directly.
    fn resolve_v_mask(&self, rmask: u8) -> u32 {
        if rmask == 0 {
            (1 << *HLEN) - 1
        } else {
            self.reg_file.v_mask()
        }
    }

    fn resolve_matrix_view(&self, slot: Option<u8>, pc: usize) -> Option<MatrixViewDescriptor> {
        slot.map(|slot| {
            self.reg_file.matrix_view(slot).unwrap_or_else(|error| {
                tracing::error!(pc, slot, %error, "invalid Matrix-view consumer");
                panic!("{error} at pc {pc}")
            })
        })
    }

    async fn read_l_tile_lines(
        &mut self,
        base: u32,
        view: MatrixViewDescriptor,
        axis: op::LTileAxis,
        lines: &[(u32, u32)],
    ) -> (QuantTensor, MatrixPacketService) {
        match axis {
            op::LTileAxis::Row => {
                self.m_machine
                    .mram
                    .read_layout_indexed_rows(base, view.layout(), lines)
                    .await
            }
            op::LTileAxis::Column => {
                self.m_machine
                    .mram
                    .read_layout_indexed_columns(base, view.layout(), lines)
                    .await
            }
        }
    }

    /// Decode the explicit Matrix-view operand marker carried by the VV
    /// family. Bits 0/1/2 select destination/source-1/source-2 slots. Keeping
    /// the marker in the instruction avoids inferring addressing semantics
    /// from whichever configuration registers happen to be live.
    fn matrix_view_operand_mask(encoded: u8) -> Option<u8> {
        (encoded & 0x8 != 0).then_some(encoded & 0x7)
    }

    async fn vector_binary_with_matrix_views(&mut self, args: MatrixViewBinaryArgs) {
        let MatrixViewBinaryArgs {
            operation,
            rd,
            rs1,
            rs2,
            rmask,
            encoded_view_mask,
            mask,
            pc,
        } = args;
        let view_mask = Self::matrix_view_operand_mask(encoded_view_mask)
            .expect("Matrix-view vector helper requires the explicit marker");
        assert_ne!(view_mask, 0, "Matrix-view operand mask cannot be zero");

        let destination_view =
            (view_mask & 0b001 != 0).then(|| self.resolve_matrix_view(Some(0), pc).unwrap());
        let source1_view =
            (view_mask & 0b010 != 0).then(|| self.resolve_matrix_view(Some(1), pc).unwrap());
        let source2_view =
            (view_mask & 0b100 != 0).then(|| self.resolve_matrix_view(Some(2), pc).unwrap());

        for descriptor in [destination_view, source1_view, source2_view]
            .into_iter()
            .flatten()
        {
            assert_eq!(
                descriptor.values(),
                self.v_machine.tile_size(),
                "a Vector Matrix-view operand must restore exactly VLEN values"
            );
        }

        let mut requests = Vec::with_capacity(2);
        if let Some(view) = source1_view {
            requests.push((self.reg_file.read_gp(rs1), view.layout()));
        }
        if let Some(view) = source2_view {
            requests.push((self.reg_file.read_gp(rs2), view.layout()));
        }
        let matrix_packets = if requests.is_empty() {
            Vec::new()
        } else {
            let (packets, service) = self.m_machine.mram.read_layout_packets(&requests).await;
            cycle!(service.service_cycles.max(1));
            packets
        };
        let mut matrix_packets = matrix_packets.into_iter();
        let lhs = if source1_view.is_some() {
            matrix_packets
                .next()
                .expect("missing Matrix source-1 packet")
        } else {
            self.v_machine.vram.read(self.reg_file.read_gp(rs1)).await
        };
        let rhs = if source2_view.is_some() {
            matrix_packets
                .next()
                .expect("missing Matrix source-2 packet")
        } else {
            self.v_machine.vram.read(self.reg_file.read_gp(rs2)).await
        };
        debug_assert!(matrix_packets.next().is_none());

        let result = self
            .v_machine
            .binary_packet(operation, lhs, rhs, rmask, mask)
            .await;
        if let Some(view) = destination_view {
            let service = self
                .m_machine
                .mram
                .write_layout_packet(self.reg_file.read_gp(rd), view.layout(), result)
                .await;
            cycle!(service.service_cycles.max(1));
        } else {
            self.v_machine
                .vram
                .write(self.reg_file.read_gp(rd), result)
                .await;
        }
    }

    /// Execute one model-independent recurrence primitive over Matrix views.
    ///
    /// Views 0/1/2 are destination/source/scalars.  The decoder owns only a
    /// deterministic row/column walk; all bases, shapes and layouts remain
    /// compiler-visible architectural state. Matrix-view storage is BF16.
    async fn execute_l_tile(&mut self, args: LTileExecArgs, pc: usize) {
        let LTileExecArgs {
            destination_register,
            source_register,
            scale_register,
            primitive,
            source_axis,
            scale_axis,
        } = args;
        let destination = self.resolve_matrix_view(Some(0), pc).unwrap();
        let source = self.resolve_matrix_view(Some(1), pc).unwrap();
        let scales = self.resolve_matrix_view(Some(2), pc).unwrap();
        let dst_base = self.reg_file.read_gp(destination_register);
        let src_base = self.reg_file.read_gp(source_register);
        let scale_base = self.reg_file.read_gp(scale_register);

        if !scales.broadcast_minor() {
            panic!("L_TILE scale view must set BROADCAST_MINOR");
        }
        if scales.shape.tile_count != 1 && scales.shape.tile_count != destination.shape.tile_count {
            panic!("L_TILE scale tiles must be one or match destination tiles");
        }
        if source.shape.tile_count != 1 && source.shape.tile_count != destination.shape.tile_count {
            panic!("L_TILE source tiles must be one or match destination tiles");
        }

        let dst_layout = destination.layout();
        let source_line_count = match source_axis {
            op::LTileAxis::Row => source.shape.rows,
            op::LTileAxis::Column => source.shape.cols,
        };
        let source_line_width = match source_axis {
            op::LTileAxis::Row => source.shape.cols,
            op::LTileAxis::Column => source.shape.rows,
        };
        let scale_line_count = match scale_axis {
            op::LTileAxis::Row => scales.shape.rows,
            op::LTileAxis::Column => scales.shape.cols,
        };
        let scale_line_width = match scale_axis {
            op::LTileAxis::Row => scales.shape.cols,
            op::LTileAxis::Column => scales.shape.rows,
        };

        match primitive {
            op::LTilePrimitive::ScaleAccum | op::LTilePrimitive::OuterUpdate => {
                if source_line_width != destination.shape.cols {
                    panic!("row-wise L_TILE source/destination widths differ");
                }
                if source_line_count != 1 && source_line_count != destination.shape.rows {
                    panic!("row-wise L_TILE source rows must be one or match destination");
                }
                if scale_line_count < destination.shape.rows {
                    panic!("L_TILE scale view has fewer logical lines than destination");
                }
                let tiles_per_packet = (self.v_machine.tile_size() / destination.shape.cols).max(1);
                for row in 0..destination.shape.rows {
                    for first_tile in
                        (0..destination.shape.tile_count).step_by(tiles_per_packet as usize)
                    {
                        let tile_count =
                            tiles_per_packet.min(destination.shape.tile_count - first_tile);
                        let destination_lines = (first_tile..first_tile + tile_count)
                            .map(|tile| (tile, row))
                            .collect::<Vec<_>>();
                        let source_lines = destination_lines
                            .iter()
                            .map(|&(tile, destination_row)| {
                                (
                                    if source.shape.tile_count == 1 {
                                        0
                                    } else {
                                        tile
                                    },
                                    if source_line_count == 1 {
                                        0
                                    } else {
                                        destination_row
                                    },
                                )
                            })
                            .collect::<Vec<_>>();
                        let scale_lines = if scales.shape.tile_count == 1 {
                            // Compact per-segment scalars are fetched once,
                            // one cycle ahead of the all-bank state packet.
                            vec![(0, row)]
                        } else {
                            destination_lines
                                .iter()
                                .map(|&(tile, destination_row)| (tile, destination_row))
                                .collect::<Vec<_>>()
                        };

                        let (dst_packet, dst_service) = self
                            .m_machine
                            .mram
                            .read_layout_indexed_rows(dst_base, dst_layout, &destination_lines)
                            .await;
                        cycle!(dst_service.service_cycles.max(1));
                        let (src_packet, src_service) = self
                            .read_l_tile_lines(src_base, source, source_axis, &source_lines)
                            .await;
                        cycle!(src_service.service_cycles.max(1));
                        let (scale_packet, scale_service) = self
                            .read_l_tile_lines(scale_base, scales, scale_axis, &scale_lines)
                            .await;
                        // Scalar bank words are deliberately charged separately:
                        // a full state packet already consumes every bank word.
                        cycle!(scale_service.service_cycles.max(1));

                        let result = match primitive {
                            op::LTilePrimitive::ScaleAccum => {
                                self.v_machine
                                    .tile_scale_accum(
                                        dst_packet,
                                        src_packet,
                                        scale_packet,
                                        destination.shape.cols,
                                        scale_line_width,
                                        first_tile,
                                    )
                                    .await
                            }
                            op::LTilePrimitive::OuterUpdate => {
                                self.v_machine
                                    .tile_outer_update(
                                        dst_packet,
                                        src_packet,
                                        scale_packet,
                                        destination.shape.cols,
                                        scale_line_width,
                                        first_tile,
                                    )
                                    .await
                            }
                            op::LTilePrimitive::DotReduce => unreachable!(),
                        };
                        let service = self
                            .m_machine
                            .mram
                            .write_layout_indexed_rows(
                                dst_base,
                                dst_layout,
                                &destination_lines,
                                result,
                            )
                            .await;
                        cycle!(service.service_cycles.max(1));
                    }
                }
            }
            op::LTilePrimitive::DotReduce => {
                if destination.shape.rows != 1
                    || destination.shape.cols != source_line_width
                    || destination.shape.tile_count != source.shape.tile_count
                {
                    panic!("DOT_REDUCE destination must be one row per source tile");
                }
                if scale_line_count < source_line_count {
                    panic!("DOT_REDUCE scale view has fewer lines than reduction rows");
                }
                let tiles_per_packet = (self.v_machine.tile_size() / source_line_width).max(1);
                for first_tile in (0..source.shape.tile_count).step_by(tiles_per_packet as usize) {
                    let tile_count = tiles_per_packet.min(source.shape.tile_count - first_tile);
                    let destination_lines = (first_tile..first_tile + tile_count)
                        .map(|tile| (tile, 0))
                        .collect::<Vec<_>>();
                    let (destination_packet, destination_service) = self
                        .m_machine
                        .mram
                        .read_layout_indexed_rows(dst_base, dst_layout, &destination_lines)
                        .await;
                    cycle!(destination_service.service_cycles.max(1));
                    let mut accumulator = tensor_to_f32_vec(destination_packet.as_tensor());
                    assert_eq!(accumulator.len(), (tile_count * source_line_width) as usize);
                    for row in 0..source_line_count {
                        let source_lines = (first_tile..first_tile + tile_count)
                            .map(|tile| (tile, row))
                            .collect::<Vec<_>>();
                        let scale_lines = if scales.shape.tile_count == 1 {
                            vec![(0, row)]
                        } else {
                            source_lines
                                .iter()
                                .map(|&(tile, source_row)| (tile, source_row))
                                .collect::<Vec<_>>()
                        };
                        let (source_packet, source_service) = self
                            .read_l_tile_lines(src_base, source, source_axis, &source_lines)
                            .await;
                        cycle!(source_service.service_cycles.max(1));
                        let (scale_packet, scale_service) = self
                            .read_l_tile_lines(scale_base, scales, scale_axis, &scale_lines)
                            .await;
                        cycle!(scale_service.service_cycles.max(1));
                        self.v_machine
                            .tile_dot_accumulate(
                                &mut accumulator,
                                source_packet,
                                scale_packet,
                                source_line_width,
                                scale_line_width,
                                first_tile,
                            )
                            .await;
                    }
                    let result = QuantTensor::quantize(
                        tensor_from_f32_slice(&accumulator),
                        self.m_machine.mram.ty(),
                    );
                    let service = self
                        .m_machine
                        .mram
                        .write_layout_indexed_rows(dst_base, dst_layout, &destination_lines, result)
                        .await;
                    cycle!(service.service_cycles.max(1));
                }
            }
        }
    }

    fn mx_region(&self, dtype: MxDataType, addr: u64, offset: u32, rstride: u8) -> dma::MxRegion {
        let scale = match dtype {
            MxDataType::Plain(_) => 0,
            MxDataType::Mx { .. } => offset / dtype.element_scale_ratio(),
        };

        dma::MxRegion {
            hbm_type: dtype,
            index: addr + offset as u64,
            // Scales are stored AFTER elements, so scale_index =
            // element_index + scale_reg + scale, where scale_reg is the offset
            // from element start to scale start.
            scale_index: addr + self.reg_file.scale() as u64 + scale as u64,
            rstride,
            stride: self.reg_file.stride(),
        }
    }

    pub(crate) async fn do_ops(
        &mut self,
        ops: &[op::Opcode],
        mut stage_profiler: Option<&mut StageProfiler>,
        mut timing: TimingDriver<'_>,
    ) {
        let mut pc: usize = 0; // Program counter

        while pc < ops.len() {
            let executed_pc = pc;
            let op = &ops[pc];

            self.loop_state.record_instruction();
            tracing::debug!(pc, ?op, "execute op");

            // L_CFG alone has no effect. Each consumer explicitly selects the
            // slots it uses; Matrix writeback uses the reserved producer slot.
            let active_lmask = self.lstream_mask_for_opcode(op);
            let stream_access = (active_lmask != 0).then(|| self.op_access_for_opcode(op));
            if let Some(access) = &stream_access {
                self.validate_lstream_opcode(op, access, active_lmask, pc);
                self.hydrate_lstream_fp_operands(access, active_lmask);
            }

            // Scoreboard mode: resolve hazards, then sleep to the modeled
            // issue instant so everything the arm does (in particular HBM
            // traffic) happens at model-consistent virtual times.
            let mut issued: Option<(Instant, OpAccess)> = None;
            if let TimingDriver::Scoreboard { scoreboard } = &mut timing {
                let access = stream_access
                    .clone()
                    .unwrap_or_else(|| self.op_access_for_opcode(op));
                // Drain policy: HBM-side ordering is not range-tracked, so
                // H_* ops order conservatively against in-flight DMAs.
                // - Barrier / H_STORE_V: everything (a store overwrites HBM an
                //   in-flight prefetch may read, and earlier stores' HBM
                //   writes may overlap its own).
                // - H_PREFETCH_*: all outstanding stores (HBM RAW) plus
                //   prefetches overlapping its SRAM destination (WAW).
                // - Everything else: SRAM overlaps only.
                let pending = if access.barrier
                    || matches!(
                        op,
                        op::Opcode::H_STORE_V { .. } | op::Opcode::H_STORE_V_MV { .. }
                    ) {
                    scoreboard.take_all_dma()
                } else if matches!(
                    op,
                    op::Opcode::H_PREFETCH_M { .. }
                        | op::Opcode::H_PREFETCH_V { .. }
                        | op::Opcode::H_PREFETCH_V_MV { .. }
                ) {
                    scoreboard.take_dma_for_prefetch(&access)
                } else {
                    scoreboard.take_overlapping_dma(&access)
                };
                for dma in pending {
                    let wait_start = Executor::current().now();
                    match dma.done.await {
                        Ok(completed_at) => scoreboard.retire_dma(&dma.writes, completed_at),
                        Err(_) => tracing::error!(
                            pc,
                            "pending DMA completer dropped its channel; timing may be optimistic"
                        ),
                    }
                    scoreboard.note_dma_wait(Executor::current().now() - wait_start);
                }
                let now = Executor::current().now();
                let issue = scoreboard.issue_bound(&access, now);
                if issue > now {
                    Executor::current().resolve_at(issue).await;
                }
                debug_assert_eq!(
                    timing::pending_charge(),
                    0,
                    "latency charge leaked across an instruction boundary"
                );
                // DMA goes asynchronous in scoreboard mode: a prefetch's fill
                // is spawned (destination cells parked Pending), a store's
                // vram rows are snapshotted here and its HBM writes spawned;
                // either way the op only occupies the DMA issue slot for one
                // cycle and its completion is carried by the registered
                // PendingDma. In serialize (serial-equivalence validation)
                // mode DMA stays inline like every other op, so the run
                // reproduces serial timing exactly.
                if !scoreboard.is_serialize() && self.issue_async_dma(op, &access, scoreboard).await
                {
                    let finish = scoreboard.commit(&access, issue, *PERIOD);
                    scoreboard.trace_op(executed_pc, op, access.unit, issue, finish);
                    if let Some(profiler) = stage_profiler.as_deref_mut() {
                        let elapsed_picos = (finish - issue).as_picos();
                        profiler.record(
                            executed_pc,
                            elapsed_picos as f64 / 1_000_000_000_000.0,
                            elapsed_picos,
                            ResourceKind::Dma,
                            // The transfer's HBM bytes land while the fill is
                            // in flight; they show up in the run-level HBM
                            // statistics, not in this op's delta.
                            0,
                            0,
                        );
                    }
                    pc += 1;
                    continue;
                }
                issued = Some((issue, access));
            }

            // Serial mode: snapshot the clock for the profiler epilogue.
            let profile_start_instant = if issued.is_none() && stage_profiler.is_some() {
                Some(Executor::current().now())
            } else {
                None
            };
            let profile_start_hbm = if stage_profiler.is_some() {
                self.hbm.statistics()
            } else {
                None
            };

            let mut jump_pc: Option<usize> = None;

            match op {
                op::Opcode::Invalid => {
                    tracing::error!(pc, "invalid opcode reached in dispatch");
                    panic!("invalid opcode at pc {pc}");
                }

                op::Opcode::M_MM { rs1, rs2, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .mm_with_view(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_MM_WO {
                    rd,
                    rstride,
                    imm,
                    view,
                } => {
                    let stride_len = if *rstride == 0 {
                        1
                    } else {
                        self.reg_file.read_gp(*rstride)
                    };
                    if let Some(view) = self.resolve_matrix_view(*view, pc) {
                        let logical_offset = if *rstride == 0 {
                            *imm
                        } else {
                            self.reg_file.read_gp(*rstride).wrapping_add(*imm)
                        };
                        let service = self
                            .m_machine
                            .mview_wo(self.reg_file.read_gp(*rd), logical_offset, view)
                            .await;
                        cycle!(service.service_cycles.max(1));
                    } else {
                        self.m_machine
                            .mm_wo(
                                self.reg_file.read_gp(*rd) + *imm,
                                stride_len,
                                self.reg_file.lstream_gp_affine_view(active_lmask, *rd),
                            )
                            .await;
                    }
                }
                op::Opcode::M_TMM { rs1, rs2, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .tmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_BMM { rs1, rs2, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .bmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_BTMM { rs1, rs2, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .btmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_BMM_WO { rd, imm } => {
                    self.m_machine
                        .bmm_wo(self.reg_file.read_gp(*rd) + *imm)
                        .await;
                }
                op::Opcode::M_MV { rs1, rs2, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .mv(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_TMV { rs1, rs2, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .tmv(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_BMV { rs1, rs2, rd, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .bmv(
                            self.reg_file.read_gp(*rs1) + self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_BTMV { rs1, rs2, rd, view } => {
                    let view = self.resolve_matrix_view(*view, pc);
                    self.m_machine
                        .btmv(
                            self.reg_file.read_gp(*rs1) + self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                            view,
                        )
                        .await;
                }
                op::Opcode::M_MV_WO { rd, imm } => {
                    self.m_machine
                        .mv_wo(self.reg_file.read_gp(*rd) + *imm)
                        .await;
                }
                op::Opcode::M_BMV_WO { rd, imm } => {
                    self.m_machine
                        .bmv_wo(self.reg_file.read_gp(*rd) + *imm)
                        .await;
                }

                op::Opcode::V_ADD_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    if Self::matrix_view_operand_mask(*lmask).is_some() {
                        self.vector_binary_with_matrix_views(MatrixViewBinaryArgs {
                            operation: VectorBinaryOp::Add,
                            rd: *rd,
                            rs1: *rs1,
                            rs2: *rs2,
                            rmask: *rmask,
                            encoded_view_mask: *lmask,
                            mask,
                            pc,
                        })
                        .await;
                    } else {
                        self.v_machine
                            .add(
                                self.reg_file.read_gp_view(*rd, *lmask),
                                self.reg_file.read_gp_view(*rs1, *lmask),
                                self.reg_file.read_gp_view(*rs2, *lmask),
                                *rmask,
                                mask,
                            )
                            .await;
                    }
                }
                op::Opcode::V_ADD_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .add_scalar(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SUB_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    if Self::matrix_view_operand_mask(*lmask).is_some() {
                        self.vector_binary_with_matrix_views(MatrixViewBinaryArgs {
                            operation: VectorBinaryOp::Sub,
                            rd: *rd,
                            rs1: *rs1,
                            rs2: *rs2,
                            rmask: *rmask,
                            encoded_view_mask: *lmask,
                            mask,
                            pc,
                        })
                        .await;
                    } else {
                        self.v_machine
                            .sub(
                                self.reg_file.read_gp_view(*rd, *lmask),
                                self.reg_file.read_gp_view(*rs1, *lmask),
                                self.reg_file.read_gp_view(*rs2, *lmask),
                                *rmask,
                                mask,
                            )
                            .await;
                    }
                }
                op::Opcode::V_SUB_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    rorder,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .sub_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                            *rorder,
                        )
                        .await;
                }
                op::Opcode::V_MUL_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    if Self::matrix_view_operand_mask(*lmask).is_some() {
                        self.vector_binary_with_matrix_views(MatrixViewBinaryArgs {
                            operation: VectorBinaryOp::Mul,
                            rd: *rd,
                            rs1: *rs1,
                            rs2: *rs2,
                            rmask: *rmask,
                            encoded_view_mask: *lmask,
                            mask,
                            pc,
                        })
                        .await;
                    } else {
                        self.v_machine
                            .mul(
                                self.reg_file.read_gp_view(*rd, *lmask),
                                self.reg_file.read_gp_view(*rs1, *lmask),
                                self.reg_file.read_gp_view(*rs2, *lmask),
                                *rmask,
                                mask,
                            )
                            .await;
                    }
                }
                op::Opcode::V_MUL_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .mul_scalar(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.vector_scalar_operand(*rs2, *lmask),
                            *rmask,
                            mask,
                            VectorOperandViews {
                                destination: self.reg_file.lstream_gp_affine_view(*lmask, *rd),
                                source: self.reg_file.lstream_gp_affine_view(*lmask, *rs1),
                            },
                        )
                        .await;
                }
                // The only V-type op that reads `rd`: `V[rd] += V[rs1] * fp[rs2]`.
                // `rd` is the destination *and* an operand, so it goes first --
                // swapping the first two arguments computes `V[rs1] += V[rd]*f`,
                // which is finite, plausible and wrong.
                op::Opcode::V_FMA_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .fma_scalar(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.vector_scalar_operand(*rs2, *lmask),
                            *rmask,
                            mask,
                            VectorOperandViews {
                                destination: self.reg_file.lstream_gp_affine_view(*lmask, *rd),
                                source: self.reg_file.lstream_gp_affine_view(*lmask, *rs1),
                            },
                        )
                        .await;
                }
                op::Opcode::V_MAX_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .max_scalar(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_MIN_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .min_scalar(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_TOPK {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let (expert_count, topk) = match *rmask {
                        0 => (32, 4),
                        1 => (128, 8),

                        15 => match self.reg_file.topk_policy() {
                            Some(policy) => policy,
                            None => {
                                tracing::error!(pc, "V_TOPK rmask=15 with no C_SET_TOPK_REG");
                                panic!(
                                    "V_TOPK rmask=15 at pc {pc} requires a preceding \
                                     C_SET_TOPK_REG; the policy register is unset"
                                );
                            }
                        },
                        other => {
                            // Consistent with the Opcode::Invalid handler: a
                            // malformed-but-encodable field is a bad-program error,
                            // logged with the pc before aborting.
                            tracing::error!(pc, rmask = other, "unsupported V_TOPK rmask policy");
                            panic!(
                                "unsupported V_TOPK rmask policy {other} at pc {pc}; \
                                 expected 0=32/top4, 1=128/top8, or 15=C_SET_TOPK_REG"
                            );
                        }
                    };
                    let fp_base = self.reg_file.read_gp(*rd) as usize;
                    let int_base = self.reg_file.read_gp(*rs2) as usize;
                    let (indices, weights) = self
                        .v_machine
                        .topk_softmax(self.reg_file.read_gp(*rs1), expert_count, topk)
                        .await;
                    for (offset, (idx, weight)) in indices.iter().zip(weights.iter()).enumerate() {
                        self.scalar_sram.write_int(int_base + offset, *idx);
                        self.scalar_sram.write_fp(fp_base + offset, *weight);
                    }
                }
                op::Opcode::V_EXP_V {
                    rd,
                    rs1,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .exp(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SOFTPLUS_V {
                    rd,
                    rs1,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .softplus(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_RECI_V {
                    rd,
                    rs1,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .reciprocal(
                            self.reg_file.read_gp_view(*rd, *lmask),
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SHFT_V { rd, rs1, rs2 } => {
                    self.v_machine
                        .shift_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                        )
                        .await;
                }
                // Write to fp0 is a no-op.
                op::Opcode::V_RED_SUM { rd: 0, .. } | op::Opcode::V_RED_MAX { rd: 0, .. } => (),

                op::Opcode::V_RED_SUM {
                    rd,
                    rs1,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    let result = self
                        .v_machine
                        .reduce_sum(
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.reg_file.read_fp(*rd).into(),
                            *rmask,
                            mask,
                            self.reg_file.lstream_gp_affine_view(*lmask, *rs1),
                        )
                        .await;
                    self.reg_file.write_fp(*rd, bf16::from_f32(result));
                }
                op::Opcode::V_RED_MAX {
                    rd,
                    rs1,
                    rmask,
                    lmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    let result = self
                        .v_machine
                        .reduce_max(
                            self.reg_file.read_gp_view(*rs1, *lmask),
                            self.reg_file.read_fp(*rd).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                    self.reg_file.write_fp(*rd, bf16::from_f32(result));
                }

                // Write to fp0 is a no-op.
                op::Opcode::S_ADD_FP { rd: 0, .. }
                | op::Opcode::S_SUB_FP { rd: 0, .. }
                | op::Opcode::S_MAX_FP { rd: 0, .. }
                | op::Opcode::S_MUL_FP { rd: 0, .. }
                | op::Opcode::S_EXP_FP { rd: 0, .. }
                | op::Opcode::S_RECI_FP { rd: 0, .. }
                | op::Opcode::S_SQRT_FP { rd: 0, .. } => {}

                op::Opcode::S_ADD_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, std::ops::Add::add);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_SUB_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, std::ops::Sub::sub);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_MAX_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, bf16::max);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_MUL_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, std::ops::Mul::mul);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_EXP_FP { rd, rs1 } => {
                    let val: f32 = self.reg_file.read_fp(*rs1).into();
                    let clamped = val.clamp(-88.0, 88.0);
                    self.reg_file.write_fp(*rd, bf16::from_f32(clamped.exp()));
                    cycle!(*SCALAR_FP_EXP_CYCLES);
                }
                op::Opcode::S_RECI_FP { rd, rs1 } => {
                    self.reg_file
                        .write_fp(*rd, bf16::ONE / self.reg_file.read_fp(*rs1));
                    cycle!(*SCALAR_FP_RECI_CYCLES);
                }
                op::Opcode::S_SQRT_FP { rd, rs1 } => {
                    self.reg_file.write_fp(
                        *rd,
                        bf16::from_f32(f32::from(self.reg_file.read_fp(*rs1)).sqrt()),
                    );
                    cycle!(*SCALAR_FP_SQRT_CYCLES);
                }
                op::Opcode::S_LD_FP { rd, rs1, imm } => {
                    self.reg_file.write_fp(
                        *rd,
                        self.scalar_sram
                            .read_fp((self.reg_file.read_gp(*rs1) + *imm) as usize),
                    );
                    cycle!(1);
                }
                op::Opcode::S_ST_FP { rd, rs1, imm } => {
                    self.scalar_sram.write_fp(
                        (self.reg_file.read_gp(*rs1) + *imm) as usize,
                        self.reg_file.read_fp(*rd),
                    );
                    cycle!(1);
                }
                op::Opcode::S_MAP_V_FP { rd, rs1, imm } => {
                    let start_idx = (self.reg_file.read_gp(*rs1) + *imm) as usize;
                    let f = self.scalar_sram.read_fp_window(start_idx, *VLEN as usize);
                    self.v_machine
                        .vector_transfer_fp(self.reg_file.read_gp(*rd), f)
                        .await;
                    cycle!(*VLEN);
                }
                op::Opcode::S_MAP_FP_V { rd, rs1, imm } => {
                    // Mirror of S_MAP_V_FP: VRAM row -> VLEN consecutive FP_MEM slots.
                    // Note the operand roles are the mirror image too: `rs1` is the
                    // VRAM source row and `rd` is the FP_MEM base, so that both
                    // instructions keep "rd names the destination memory".
                    let values = self
                        .v_machine
                        .vector_read_fp(self.reg_file.read_gp(*rs1))
                        .await;
                    let start_idx = (self.reg_file.read_gp(*rd) + *imm) as usize;
                    self.scalar_sram.write_fp_window(start_idx, &values);
                    cycle!(*VLEN);
                }
                op::Opcode::S_ADD_INT { rd, rs1, rs2 } => {
                    self.reg_file.binop_gp(*rd, *rs1, *rs2, u32::wrapping_add);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ADDI_INT { rd, rs1, imm } => {
                    self.reg_file
                        .write_gp(*rd, self.reg_file.read_gp(*rs1).wrapping_add(*imm));
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_SUB_INT { rd, rs1, rs2 } => {
                    self.reg_file.binop_gp(*rd, *rs1, *rs2, u32::wrapping_sub);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_MUL_INT { rd, rs1, rs2 } => {
                    self.reg_file.binop_gp(*rd, *rs1, *rs2, u32::wrapping_mul);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LUI_INT { rd, imm } => {
                    self.reg_file.write_gp(*rd, *imm << 12);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LD_INT { rd, rs1, imm } => {
                    self.reg_file.write_gp(
                        *rd,
                        self.scalar_sram
                            .read_int((self.reg_file.read_gp(*rs1) + *imm) as usize),
                    );
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ST_INT { rd, rs1, imm } => {
                    self.scalar_sram.write_int(
                        (self.reg_file.read_gp(*rs1) + *imm) as usize,
                        self.reg_file.read_gp(*rd),
                    );
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::H_PREFETCH_M {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    // TODO: rstride support to be added
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::MatrixPrecision::Weights => *MATRIX_WEIGHT_TYPE,
                        op::MatrixPrecision::KeyValue => *MATRIX_KV_TYPE,
                    };

                    let region = self.mx_region(dtype, addr, offset, *rstride);
                    let xfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.m_machine.mram.ty(),
                        *MLEN,
                        *PREFETCH_M_AMOUNT,
                        *MLEN,
                    );

                    self.m_machine
                        .mram
                        .continous_write_delayed(
                            self.reg_file.read_gp(*rd),
                            *PREFETCH_M_AMOUNT,
                            xfer,
                        )
                        .await;
                }
                op::Opcode::H_PREFETCH_V {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    // TODO: rstride support to be added
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                        op::VectorPrecision::State => *STATE_TYPE,
                    };

                    let region = self.mx_region(dtype, addr, offset, *rstride);
                    let xfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.v_machine.vram.ty(),
                        *VLEN,
                        *PREFETCH_V_AMOUNT,
                        1,
                    );

                    let dest = self.reg_file.read_gp(*rd);
                    self.v_machine
                        .vram
                        .continous_write_delayed(dest, *PREFETCH_V_AMOUNT, xfer)
                        .await;
                }
                op::Opcode::H_PREFETCH_V_MV {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                    view,
                } => {
                    let descriptor = self.resolve_matrix_view(Some(*view), pc).unwrap();
                    let values = descriptor.values();
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                        op::VectorPrecision::State => *STATE_TYPE,
                    };
                    let region = self.mx_region(
                        dtype,
                        self.reg_file.read_hbm(*rs2),
                        self.reg_file.read_gp(*rs1),
                        *rstride,
                    );
                    let xfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.m_machine.mram.ty(),
                        *VLEN,
                        values.div_ceil(*VLEN),
                        1,
                    );
                    let tensor = xfer.await.unwrap_or_else(|error| {
                        panic!("Matrix-view DMA receiver dropped: {error}")
                    });
                    let tensor = if tensor.as_tensor().numel() == values as usize {
                        tensor
                    } else {
                        QuantTensor::quantize(
                            tensor.as_tensor().narrow(0, 0, i64::from(values)),
                            self.m_machine.mram.ty(),
                        )
                    };
                    let service = self
                        .m_machine
                        .mram
                        .write_layout_packet(
                            self.reg_file.read_gp(*rd),
                            descriptor.layout(),
                            tensor,
                        )
                        .await;
                    cycle!(service.service_cycles.max(1));
                }
                op::Opcode::H_STORE_V {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    let src_addr = self.reg_file.read_gp(*rd);
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                        op::VectorPrecision::State => *STATE_TYPE,
                    };

                    let region = self.mx_region(dtype, addr, offset, *rstride);

                    dma::transfer_mx_to_hbm(
                        &self.hbm,
                        &self.v_machine.vram,
                        region,
                        src_addr,
                        *VLEN,
                        *STORE_V_AMOUNT,
                    )
                    .await;
                }
                op::Opcode::H_STORE_V_MV {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                    view,
                } => {
                    let descriptor = self.resolve_matrix_view(Some(*view), pc).unwrap();
                    let (packet, service) = self
                        .m_machine
                        .mram
                        .read_layout_packet(self.reg_file.read_gp(*rd), descriptor.layout())
                        .await;
                    cycle!(service.service_cycles.max(1));
                    let rows = dma::split_packet_rows(&packet, *VLEN, self.m_machine.mram.ty());
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                        op::VectorPrecision::State => *STATE_TYPE,
                    };
                    let region = self.mx_region(
                        dtype,
                        self.reg_file.read_hbm(*rs2),
                        self.reg_file.read_gp(*rs1),
                        *rstride,
                    );
                    dma::store_rows_to_hbm(&self.hbm, region, rows, *VLEN).await;
                }
                op::Opcode::C_SET_ADDR_REG { rd, rs1, rs2 } => {
                    let imm = ((self.reg_file.read_gp(*rs1) as u64) << 32)
                        | (self.reg_file.read_gp(*rs2) as u64);
                    self.reg_file.write_hbm(*rd, imm);
                    cycle!(1);
                }
                op::Opcode::C_SET_SCALE_REG { rd } => {
                    self.reg_file.set_scale(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::C_SET_STRIDE_REG { rd } => {
                    self.reg_file.set_stride(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::C_SET_V_MASK_REG { rd } => {
                    self.reg_file.set_v_mask(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::C_SET_TOPK_REG { rd } => {
                    self.reg_file.set_topk_policy(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::L_CFG {
                    value,
                    target,
                    slot,
                    field,
                } => {
                    let field = ConfigField::try_from(*field).unwrap_or_else(|error| {
                        tracing::error!(pc, %error, "invalid L_CFG field");
                        panic!("{error} at pc {pc}");
                    });
                    let value = self.reg_file.read_gp(*value);
                    self.reg_file
                        .configure_lstream(value, *target, *slot, field)
                        .unwrap_or_else(|error| {
                            tracing::error!(pc, %error, "invalid L_CFG value");
                            panic!("{error} at pc {pc}");
                        });
                    cycle!(1);
                }
                op::Opcode::L_TILE_CFG {
                    shape,
                    mapping,
                    slot,
                } => {
                    self.reg_file
                        .configure_mview(*slot, *shape, *mapping)
                        .unwrap_or_else(|error| {
                            tracing::error!(pc, slot, %error, "invalid L_TILE_CFG");
                            panic!("{error} at pc {pc}")
                        });
                    cycle!(1);
                }
                op::Opcode::L_TILE_EXEC {
                    rd,
                    rs1,
                    rs2,
                    primitive,
                    source_axis,
                    scale_axis,
                } => {
                    self.execute_l_tile(
                        LTileExecArgs {
                            destination_register: *rd,
                            source_register: *rs1,
                            scale_register: *rs2,
                            primitive: *primitive,
                            source_axis: *source_axis,
                            scale_axis: *scale_axis,
                        },
                        pc,
                    )
                    .await;
                }
                op::Opcode::C_LOOP_START { rd, imm } => {
                    self.loop_state.start(pc, *rd, *imm, &mut self.reg_file);
                    cycle!(1);
                }
                op::Opcode::C_LOOP_END { rd } => {
                    if let LoopDecision::JumpTo(target_pc) =
                        self.loop_state.end(*rd, &mut self.reg_file)
                    {
                        jump_pc = Some(target_pc);
                    }
                    cycle!(1);
                }
                op::Opcode::C_BREAK => {
                    self.loop_state.break_innermost(&mut self.reg_file);
                    cycle!(1);
                }
            }

            if stream_access.is_some() {
                self.reg_file.advance_lstream_mask(active_lmask);
            }

            // Handle loop jumps
            if let Some(target_pc) = jump_pc {
                pc = target_pc;
            } else {
                pc += 1;
            }

            // Epilogue: commit/charge modeled time and feed the profilers.
            let mut profiled_elapsed_picos: Option<u64> = None;
            match &mut timing {
                TimingDriver::Scoreboard { scoreboard } => {
                    let (issue, access) = issued.take().expect("scoreboard pre-issue ran");
                    let charged = timing::take_charged();
                    let after = Executor::current().now();
                    let is_dma_op = matches!(
                        op,
                        op::Opcode::H_PREFETCH_M { .. }
                            | op::Opcode::H_PREFETCH_V { .. }
                            | op::Opcode::H_STORE_V { .. }
                    );
                    if after > issue && !is_dma_op {
                        tracing::warn!(
                            pc = executed_pc,
                            ?op,
                            "unclassified dependency stalled dispatch past its modeled issue"
                        );
                    }
                    // Inline HBM transfers advance the clock themselves
                    // (`after - issue`); everything else charges through
                    // `timing::charge_cycles`.
                    let latency = (after - issue) + *PERIOD * charged;
                    let finish = scoreboard.commit(&access, issue, latency);
                    scoreboard.trace_op(executed_pc, op, access.unit, issue, finish);
                    profiled_elapsed_picos = Some((finish - issue).as_picos());
                }
                TimingDriver::Serial => {
                    if let Some(start_instant) = profile_start_instant {
                        profiled_elapsed_picos =
                            Some((Executor::current().now() - start_instant).as_picos());
                    }
                }
            }
            if let (Some(elapsed_picos), Some(profiler)) =
                (profiled_elapsed_picos, stage_profiler.as_deref_mut())
            {
                let elapsed_secs = elapsed_picos as f64 / 1_000_000_000_000.0;
                let (hbm_bytes_read, hbm_bytes_written) = if let (Some(before), Some(after)) =
                    (profile_start_hbm, self.hbm.statistics())
                {
                    (
                        after
                            .total_bytes_read
                            .saturating_sub(before.total_bytes_read),
                        after
                            .total_bytes_written
                            .saturating_sub(before.total_bytes_written),
                    )
                } else {
                    (0, 0)
                };
                profiler.record(
                    executed_pc,
                    elapsed_secs,
                    elapsed_picos,
                    resource_kind_for_opcode(op),
                    hbm_bytes_read,
                    hbm_bytes_written,
                );
            }
        }

        // End of program in scoreboard mode: land every in-flight DMA, then
        // advance the clock to the modeled finish so `executor.now()` (the
        // reported total) reflects pipelined completion.
        if let TimingDriver::Scoreboard { scoreboard } = &mut timing {
            for dma in scoreboard.take_all_dma() {
                match dma.done.await {
                    Ok(completed_at) => scoreboard.retire_dma(&dma.writes, completed_at),
                    Err(_) => tracing::error!(
                        "pending DMA completer dropped its channel at end of program"
                    ),
                }
            }
            let now = Executor::current().now();
            let finish = scoreboard.max_finish().max(now);
            if finish > now {
                Executor::current().resolve_at(finish).await;
            }
        }
    }

    fn op_access_for_opcode(&self, op: &op::Opcode) -> OpAccess {
        if let op::Opcode::H_PREFETCH_V_MV { view, .. } | op::Opcode::H_STORE_V_MV { view, .. } = op
        {
            let descriptor = self.reg_file.matrix_view(*view).unwrap_or_else(|error| {
                panic!("{error} while building Matrix-view DMA scoreboard access")
            });
            let mut dma_access = access::op_access(op, &|reg| self.reg_file.read_gp(reg), &|| {
                self.reg_file.topk_policy()
            });
            for resource in dma_access
                .reads
                .iter_mut()
                .chain(dma_access.writes.iter_mut())
            {
                if let access::Resource::Sram(range) = resource
                    && range.space == access::SramSpace::Matrix
                {
                    range.len = descriptor.values();
                }
            }
            return dma_access;
        }

        let matrix_vector = match op {
            op::Opcode::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask,
                lmask,
            }
            | op::Opcode::V_SUB_VV {
                rd,
                rs1,
                rs2,
                rmask,
                lmask,
            }
            | op::Opcode::V_MUL_VV {
                rd,
                rs1,
                rs2,
                rmask,
                lmask,
            } if Self::matrix_view_operand_mask(*lmask).is_some() => {
                Some((*rd, *rs1, *rs2, *rmask, *lmask))
            }
            _ => None,
        };
        if let Some((rd, rs1, rs2, rmask, encoded_mask)) = matrix_vector {
            use access::{Cfg, Resource, SramRange, SramSpace, Unit};

            let view_mask = Self::matrix_view_operand_mask(encoded_mask).unwrap();
            let mut reads = vec![
                Resource::Gp(rd),
                Resource::Gp(rs1),
                Resource::Gp(rs2),
                Resource::Cfg(Cfg::MatrixView),
            ];
            if rmask != 0 {
                reads.push(Resource::Cfg(Cfg::VMask));
            }
            for (slot, register) in [(1_u8, rs1), (2_u8, rs2)] {
                let (space, len) = if view_mask & (1 << slot) != 0 {
                    let view = self.reg_file.matrix_view(slot).unwrap_or_else(|error| {
                        panic!("{error} while building Matrix-view scoreboard access")
                    });
                    (SramSpace::Matrix, view.values())
                } else {
                    (SramSpace::Vector, *VLEN)
                };
                reads.push(Resource::Sram(SramRange::new(
                    space,
                    self.reg_file.read_gp(register),
                    len,
                )));
            }
            let (space, len) = if view_mask & 0b001 != 0 {
                let view = self.reg_file.matrix_view(0).unwrap_or_else(|error| {
                    panic!("{error} while building Matrix-view scoreboard access")
                });
                (SramSpace::Matrix, view.values())
            } else {
                (SramSpace::Vector, *VLEN)
            };
            return OpAccess {
                unit: Unit::Vector,
                barrier: false,
                reads,
                writes: vec![Resource::Sram(SramRange::new(
                    space,
                    self.reg_file.read_gp(rd),
                    len,
                ))],
            };
        }

        let lmask = self.lstream_mask_for_opcode(op);
        // Matrix writeback keeps its compiler-written logical pointer; slot 3
        // changes physical placement only. Consumer views replace addresses.
        let producer = matches!(op, op::Opcode::M_MM_WO { .. });
        let mut access = access::op_access(
            op,
            &|reg| {
                if producer {
                    self.reg_file.read_gp(reg)
                } else {
                    self.reg_file.read_gp_view(reg, lmask)
                }
            },
            &|| self.reg_file.topk_policy(),
        );
        if lmask != 0 {
            access
                .reads
                .push(access::Resource::Cfg(access::Cfg::LStream));
        }
        access
    }

    fn hydrate_lstream_fp_operands(&mut self, access: &OpAccess, lmask: u8) {
        let mut registers = std::collections::BTreeSet::new();
        for resource in &access.reads {
            if let access::Resource::Fp(register) = resource {
                registers.insert(*register);
            }
        }
        for register in registers {
            if let Some(address) = self.reg_file.lstream_fp_address(lmask, register) {
                let value = self.scalar_sram.read_fp(address as usize);
                self.reg_file.write_fp(register, value);
            }
        }
    }

    fn vector_scalar_operand(&self, register: u8, lmask: u8) -> ScalarOperand {
        if let Some(packet) = self.reg_file.lstream_fp_packet(lmask, register) {
            assert_eq!(
                packet.packet_elements,
                self.v_machine.tile_size(),
                "segmented scalar packet must expand to VLEN elements"
            );
            assert_eq!(
                packet.packet_elements % packet.storage_atom,
                0,
                "segmented scalar packet must contain whole atoms"
            );
            let segments = packet.packet_elements / packet.storage_atom;
            let values = (0..segments)
                .map(|segment| {
                    let address = packet
                        .origin
                        .checked_add(segment * packet.packet_stride)
                        .expect("segmented scalar packet address overflow");
                    f32::from(self.scalar_sram.read_fp(address as usize))
                })
                .collect();
            ScalarOperand::Segmented {
                values,
                storage_atom: packet.storage_atom,
            }
        } else {
            ScalarOperand::Broadcast(self.reg_file.read_fp(register).into())
        }
    }

    fn validate_lstream_opcode(&self, op: &op::Opcode, access: &OpAccess, lmask: u8, pc: usize) {
        if let op::Opcode::M_MM_WO { rd, .. } = op {
            self.reg_file
                .validate_lstream_producer_mask(lmask, *rd)
                .unwrap_or_else(|error| {
                    panic!("invalid L-Compute producer view at pc {pc}: {error}")
                });
        } else {
            let targets = access
                .reads
                .iter()
                .chain(&access.writes)
                .filter_map(|resource| match resource {
                    access::Resource::Gp(register) => Some(StreamTarget::Gp(*register)),
                    access::Resource::Fp(register) => Some(StreamTarget::Fp(*register)),
                    _ => None,
                });
            self.reg_file
                .validate_lstream_mask(lmask, targets)
                .unwrap_or_else(|error| panic!("invalid L-Compute view at pc {pc}: {error}"));
        }

        let affine_targets: Vec<_> = access
            .reads
            .iter()
            .filter_map(|resource| match resource {
                access::Resource::Gp(register)
                    if self
                        .reg_file
                        .lstream_gp_affine_view(lmask, *register)
                        .is_some() =>
                {
                    Some(*register)
                }
                _ => None,
            })
            .collect();
        if affine_targets.is_empty() {
            return;
        }
        if matches!(
            op,
            op::Opcode::M_MM_WO { .. }
                | op::Opcode::V_MUL_VF { .. }
                | op::Opcode::V_FMA_VF { .. }
                | op::Opcode::V_RED_SUM { .. }
        ) {
            return;
        }
        panic!(
            "affine L-stream targets {affine_targets:?} on unsupported opcode {op:?} at pc {pc}; \
             use the identity stream or the static fallback"
        );
    }

    fn lstream_mask_for_opcode(&self, op: &op::Opcode) -> u8 {
        match op {
            op::Opcode::V_ADD_VV { lmask, .. }
            | op::Opcode::V_SUB_VV { lmask, .. }
            | op::Opcode::V_MUL_VV { lmask, .. }
                if Self::matrix_view_operand_mask(*lmask).is_some() =>
            {
                0
            }
            op::Opcode::V_ADD_VV { lmask, .. }
            | op::Opcode::V_ADD_VF { lmask, .. }
            | op::Opcode::V_SUB_VV { lmask, .. }
            | op::Opcode::V_MUL_VV { lmask, .. }
            | op::Opcode::V_MUL_VF { lmask, .. }
            | op::Opcode::V_FMA_VF { lmask, .. }
            | op::Opcode::V_MAX_VF { lmask, .. }
            | op::Opcode::V_MIN_VF { lmask, .. }
            | op::Opcode::V_EXP_V { lmask, .. }
            | op::Opcode::V_RECI_V { lmask, .. }
            | op::Opcode::V_RED_SUM { lmask, .. }
            | op::Opcode::V_RED_MAX { lmask, .. }
            | op::Opcode::V_SOFTPLUS_V { lmask, .. } => *lmask,
            // A view-qualified writeback uses only its explicit Matrix-view
            // descriptor and logical-offset register.  A stale legacy L_CFG
            // binding on the same GP register must not activate stream
            // addressing or advance hidden stream state.
            op::Opcode::M_MM_WO { view: Some(_), .. } => 0,
            op::Opcode::M_MM_WO { rd, view: None, .. } => self.reg_file.lstream_producer_mask(*rd),
            _ => 0,
        }
    }

    /// Scoreboard-mode asynchronous DMA issue: launch the HBM traffic at the
    /// current (modeled issue) instant and spawn a completer that reports the
    /// real completion instant. For a prefetch, the destination SRAM cells
    /// are parked as `Cell::Pending` first; for a store, the source vram rows
    /// are snapshotted here (functional WAR safety) before the HBM writes are
    /// spawned. Returns `false` for any other opcode, leaving it to the
    /// normal inline path.
    ///
    /// The `Cell::Pending` parking is the functional safety net: even a
    /// dependency the access descriptor misses still blocks on first read
    /// instead of observing stale data.
    async fn issue_async_dma(
        &mut self,
        op: &op::Opcode,
        access: &OpAccess,
        scoreboard: &mut Scoreboard,
    ) -> bool {
        let (kind, done_rx) = match op {
            op::Opcode::H_PREFETCH_M {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
                let offset = self.reg_file.read_gp(*rs1);
                let addr = self.reg_file.read_hbm(*rs2);
                let dtype = match precision {
                    op::MatrixPrecision::Weights => *MATRIX_WEIGHT_TYPE,
                    op::MatrixPrecision::KeyValue => *MATRIX_KV_TYPE,
                };
                let region = self.mx_region(dtype, addr, offset, *rstride);
                let xfer = dma::transfer_mx_from_hbm(
                    &self.hbm,
                    region,
                    self.m_machine.mram.ty(),
                    *MLEN,
                    *PREFETCH_M_AMOUNT,
                    *MLEN,
                );
                let dest = self.reg_file.read_gp(*rd);
                // The transfer produces MLEN * PREFETCH_M_AMOUNT elements.
                let cells = (*MLEN * *PREFETCH_M_AMOUNT).div_ceil(*MLEN * *MLEN);
                let senders = self.m_machine.mram.mark_pending_tiles(dest, cells).await;
                let mram = self.m_machine.mram.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    mram.fill_pending(senders, xfer).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Prefetch, done_rx)
            }
            op::Opcode::H_PREFETCH_V {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
                let offset = self.reg_file.read_gp(*rs1);
                let addr = self.reg_file.read_hbm(*rs2);
                let dtype = match precision {
                    op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                    op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    op::VectorPrecision::State => *STATE_TYPE,
                };
                let region = self.mx_region(dtype, addr, offset, *rstride);
                let xfer = dma::transfer_mx_from_hbm(
                    &self.hbm,
                    region,
                    self.v_machine.vram.ty(),
                    *VLEN,
                    *PREFETCH_V_AMOUNT,
                    1,
                );
                let dest = self.reg_file.read_gp(*rd);
                let senders = self
                    .v_machine
                    .vram
                    .mark_pending_rows(dest, *PREFETCH_V_AMOUNT)
                    .await;
                let vram = self.v_machine.vram.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    vram.fill_pending(senders, xfer).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Prefetch, done_rx)
            }
            op::Opcode::H_PREFETCH_V_MV {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
                view,
            } => {
                let descriptor = self
                    .reg_file
                    .matrix_view(*view)
                    .unwrap_or_else(|error| panic!("{error} while issuing Matrix-view prefetch"));
                let values = descriptor.values();
                let dtype = match precision {
                    op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                    op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    op::VectorPrecision::State => *STATE_TYPE,
                };
                let region = self.mx_region(
                    dtype,
                    self.reg_file.read_hbm(*rs2),
                    self.reg_file.read_gp(*rs1),
                    *rstride,
                );
                let xfer = dma::transfer_mx_from_hbm(
                    &self.hbm,
                    region,
                    self.m_machine.mram.ty(),
                    *VLEN,
                    values.div_ceil(*VLEN),
                    1,
                );
                let dest = self.reg_file.read_gp(*rd);
                let (pending, service) = self
                    .m_machine
                    .mram
                    .mark_pending_layout_packet(dest, descriptor.layout())
                    .await;
                let mram = self.m_machine.mram.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    let tensor = xfer.await.unwrap_or_else(|error| {
                        panic!("Matrix-view DMA receiver dropped: {error}")
                    });
                    Executor::current()
                        .resolve_at(*PERIOD * service.service_cycles.max(1))
                        .await;
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    let _ = tx.send(tensor);
                    mram.fill_pending(pending, rx).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Prefetch, done_rx)
            }
            op::Opcode::H_STORE_V {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
                let src_addr = self.reg_file.read_gp(*rd);
                let offset = self.reg_file.read_gp(*rs1);
                let addr = self.reg_file.read_hbm(*rs2);
                let dtype = match precision {
                    op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                    op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    op::VectorPrecision::State => *STATE_TYPE,
                };
                let region = self.mx_region(dtype, addr, offset, *rstride);
                // Snapshot the source rows at issue: later instructions may
                // overwrite them while the HBM writes are still in flight.
                let rows =
                    dma::snapshot_vram_rows(&self.v_machine.vram, src_addr, *VLEN, *STORE_V_AMOUNT)
                        .await;
                let hbm = self.hbm.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    dma::store_rows_to_hbm(&hbm, region, rows, *VLEN).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Store, done_rx)
            }
            op::Opcode::H_STORE_V_MV {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
                view,
            } => {
                let descriptor = self
                    .reg_file
                    .matrix_view(*view)
                    .unwrap_or_else(|error| panic!("{error} while issuing Matrix-view store"));
                let (packet, service) = self
                    .m_machine
                    .mram
                    .read_layout_packet(self.reg_file.read_gp(*rd), descriptor.layout())
                    .await;
                let rows = dma::split_packet_rows(&packet, *VLEN, self.m_machine.mram.ty());
                let dtype = match precision {
                    op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                    op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    op::VectorPrecision::State => *STATE_TYPE,
                };
                let region = self.mx_region(
                    dtype,
                    self.reg_file.read_hbm(*rs2),
                    self.reg_file.read_gp(*rs1),
                    *rstride,
                );
                let hbm = self.hbm.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    Executor::current()
                        .resolve_at(*PERIOD * service.service_cycles.max(1))
                        .await;
                    dma::store_rows_to_hbm(&hbm, region, rows, *VLEN).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Store, done_rx)
            }
            _ => return false,
        };
        let writes: Vec<access::SramRange> = access
            .writes
            .iter()
            .filter_map(|r| match r {
                access::Resource::Sram(range) => Some(*range),
                _ => None,
            })
            .collect();
        scoreboard.register_dma(kind, writes, done_rx);
        true
    }
}

fn resource_kind_for_opcode(op: &op::Opcode) -> ResourceKind {
    match op {
        op::Opcode::M_MM { .. }
        | op::Opcode::M_TMM { .. }
        | op::Opcode::M_BMM { .. }
        | op::Opcode::M_BTMM { .. }
        | op::Opcode::M_BMM_WO { .. }
        | op::Opcode::M_MM_WO { .. }
        | op::Opcode::M_MV { .. }
        | op::Opcode::M_TMV { .. }
        | op::Opcode::M_BMV { .. }
        | op::Opcode::M_BTMV { .. }
        | op::Opcode::M_MV_WO { .. }
        | op::Opcode::M_BMV_WO { .. } => ResourceKind::Matrix,

        op::Opcode::V_ADD_VV { .. }
        | op::Opcode::V_ADD_VF { .. }
        | op::Opcode::V_SUB_VV { .. }
        | op::Opcode::V_SUB_VF { .. }
        | op::Opcode::V_MUL_VV { .. }
        | op::Opcode::V_MUL_VF { .. }
        | op::Opcode::V_MAX_VF { .. }
        | op::Opcode::V_MIN_VF { .. }
        | op::Opcode::V_TOPK { .. }
        | op::Opcode::V_EXP_V { .. }
        | op::Opcode::V_RECI_V { .. }
        | op::Opcode::V_RED_SUM { .. }
        | op::Opcode::V_RED_MAX { .. }
        | op::Opcode::V_FMA_VF { .. }
        | op::Opcode::V_SOFTPLUS_V { .. }
        // L_TILE_EXEC sequences existing Vector mul/add/reduce datapaths over
        // Matrix-view packets.  Its configuration is scalar, but its execution
        // must be charged to the arithmetic resource it occupies.
        | op::Opcode::L_TILE_EXEC { .. }
        // S_MAP_FP_V drives the vector SRAM read port for a whole VLEN row, so it
        // contends with the vector unit even though its destination is FP_MEM. Its
        // mirror S_MAP_V_FP is classified Scalar for the same reason inverted.
        | op::Opcode::S_MAP_FP_V { .. }
        | op::Opcode::V_SHFT_V { .. } => ResourceKind::Vector,

        op::Opcode::S_ADD_FP { .. }
        | op::Opcode::S_SUB_FP { .. }
        | op::Opcode::S_MAX_FP { .. }
        | op::Opcode::S_MUL_FP { .. }
        | op::Opcode::S_EXP_FP { .. }
        | op::Opcode::S_RECI_FP { .. }
        | op::Opcode::S_SQRT_FP { .. }
        | op::Opcode::S_LD_FP { .. }
        | op::Opcode::S_ST_FP { .. }
        | op::Opcode::S_MAP_V_FP { .. }
        | op::Opcode::S_ADD_INT { .. }
        | op::Opcode::S_ADDI_INT { .. }
        | op::Opcode::S_SUB_INT { .. }
        | op::Opcode::S_MUL_INT { .. }
        | op::Opcode::S_LUI_INT { .. }
        | op::Opcode::S_LD_INT { .. }
        | op::Opcode::S_ST_INT { .. }
        | op::Opcode::C_SET_ADDR_REG { .. }
        | op::Opcode::C_SET_SCALE_REG { .. }
        | op::Opcode::C_SET_STRIDE_REG { .. }
        | op::Opcode::C_SET_V_MASK_REG { .. }
        | op::Opcode::C_SET_TOPK_REG { .. }
        | op::Opcode::L_CFG { .. }
        | op::Opcode::L_TILE_CFG { .. }
        | op::Opcode::C_LOOP_START { .. }
        | op::Opcode::C_LOOP_END { .. }
        | op::Opcode::C_BREAK => ResourceKind::Scalar,

        op::Opcode::H_PREFETCH_M { .. }
        | op::Opcode::H_PREFETCH_V { .. }
        | op::Opcode::H_PREFETCH_V_MV { .. }
        | op::Opcode::H_STORE_V { .. }
        | op::Opcode::H_STORE_V_MV { .. } => ResourceKind::Dma,

        op::Opcode::Invalid => ResourceKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `V_FMA_VF` belongs to the vector unit, not the scalar one.
    #[test]
    fn fma_is_billed_to_the_vector_unit() {
        assert_eq!(
            resource_kind_for_opcode(&op::Opcode::V_FMA_VF {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
                lmask: 0,
            }),
            ResourceKind::Vector
        );
    }

    #[test]
    fn l_tile_execution_is_billed_to_the_vector_unit() {
        assert_eq!(
            resource_kind_for_opcode(&op::Opcode::L_TILE_EXEC {
                rd: 1,
                rs1: 2,
                rs2: 3,
                primitive: op::LTilePrimitive::ScaleAccum,
                source_axis: op::LTileAxis::Row,
                scale_axis: op::LTileAxis::Row,
            }),
            ResourceKind::Vector
        );
    }
}
