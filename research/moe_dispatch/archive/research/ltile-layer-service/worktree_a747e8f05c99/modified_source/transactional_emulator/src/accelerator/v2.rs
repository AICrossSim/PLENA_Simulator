//! V2 reduction/update execution. Persistent state is always the banked
//! Matrix SRAM. A tree uses Vector SRAM rows 0..6 and leaf/output row 7;
//! only the optional FP32 implementation allocates a P-element context.
use super::{Accelerator, dispatch::LTileExecArgs, mview::MatrixViewDescriptor};
use crate::{
    ltile_v2::round_bf16,
    op::{LTileAxis, LTilePrimitive as Op},
    v2_timing::Calendar,
};
use quantize::{QuantTensor, tensor_from_f32_slice, tensor_to_f32_vec};

#[derive(Default)]
pub(super) struct ReductionState {
    active: bool,
    values: usize,
    rows: u32,
    sums: Vec<f32>,
}
fn number(name: &str, fallback: u32) -> u32 {
    let n = std::env::var(name).map_or(fallback, |s| s.parse().expect("invalid v2 parameter"));
    assert!(n > 0);
    n
}
fn tree() -> bool {
    match std::env::var("PLENA_V2_DOT").as_deref() {
        Ok("tree") => true,
        Ok("fp32") | Err(_) => false,
        _ => panic!("PLENA_V2_DOT must be tree or fp32"),
    }
}
fn rn(value: f32) -> f32 {
    f32::from_bits(u32::from(round_bf16(value, None)) << 16)
}

impl Accelerator {
    pub(crate) fn v2_config_report(&self) -> serde_json::Value {
        let c = self.v_machine.update_lane.config;
        serde_json::json!({
            "L": c.lanes, "update_latency": c.latency, "update_II": c.interval,
            "dot": if tree() { "BF16_SRAM_tree" } else { "FP32_context" },
            "native_coefficients": {
                "descriptors": "3 x 64 bit; L_TILE form2, native primitives9..11",
                "state_read_width": number("PLENA_NATIVE_READ_WIDTH", 512),
                "result_slots": number("PLENA_NATIVE_RESULT_SLOTS", 4),
                "sector_limit_bytes": 4096, "operand_latch_bytes": 4096,
                "address_generation": "one coefficient word address/cycle",
                "alignment_cycles": 1, "selection_cycles": 1,
                "prefetch": "two bounded state slots, released only after last operand acceptance",
            },
            "P": 2048, "allocated_context_bytes": if tree() { 0 } else { 8192 },
            "tree_rows": if tree() { 8 } else { 0 },
            "feedback_core_latency": number("PLENA_V2_DOT_LATENCY", 6),
            "dot_II": number("PLENA_V2_DOT_II", 2),
            "context_read_write_cycles_each": number("PLENA_V2_CONTEXT_PORT", 1),
            "sram_port_multiplier": number("PLENA_V2_SRAM_PORT", 1),
            "state_rounding": if c.stochastic { "per_lane_LFSR_SR" } else { "BF16_RNE" },
            "seed": c.seed,
            "unvalidated_RTL": "FP32 decayed-state-to-product forwarding, dot/residual path, context storage/ports and vector bank-write enables; only update 6-cycle II2 lane has a D probe",
            "ports": "one Matrix view packet at a time; independent one-port Vector SRAM; FP32 context one L-wide read and write per cycle, fixed banks",
            "active_context_feedback": "actual slot ready times, not allocated P/L",
        })
    }

    async fn v2_read(
        &mut self,
        base: u32,
        view: MatrixViewDescriptor,
        lines: &[(u32, u32)],
        ready: u64,
        clock: &mut Calendar,
    ) -> (Vec<f32>, u64) {
        let (q, s) = self
            .m_machine
            .mram
            .read_layout_indexed_rows(base, view.layout(), lines)
            .await;
        let end = clock.matrix(
            ready,
            s.service_cycles * u64::from(number("PLENA_V2_SRAM_PORT", 1)),
            s.bank_words,
            false,
        );
        (tensor_to_f32_vec(q.as_tensor()), end)
    }

    async fn v2_write(
        &mut self,
        base: u32,
        view: MatrixViewDescriptor,
        lines: &[(u32, u32)],
        values: &[f32],
        ready: u64,
        clock: &mut Calendar,
    ) -> u64 {
        let q = QuantTensor::quantize(tensor_from_f32_slice(values), self.m_machine.mram.ty());
        let s = self
            .m_machine
            .mram
            .write_layout_indexed_rows(base, view.layout(), lines, q)
            .await;
        clock.matrix(
            ready,
            s.service_cycles * u64::from(number("PLENA_V2_SRAM_PORT", 1)),
            s.bank_words,
            true,
        )
    }

    async fn tree_values(&mut self, row: u32) -> Vec<f32> {
        let q = self
            .v_machine
            .vram
            .read(row * self.v_machine.tile_size())
            .await;
        tensor_to_f32_vec(q.as_tensor())
    }

    async fn tree_write(&mut self, row: u32, first: u32, values: &[f32]) {
        let q = QuantTensor::quantize(tensor_from_f32_slice(values), self.v_machine.vram.ty());
        self.v_machine
            .vram
            .write_bank_words(row * self.v_machine.tile_size(), first, q)
            .await;
    }

    /// Fold one complete leaf row with the existing full-width BF16 Vector
    /// adder. Do not serialize this operation into L-wide FP32 lane work.
    async fn tree_fold(&mut self, count: u32, ready: u64, clock: &mut Calendar) -> u64 {
        let mut ready = ready;
        let port = u64::from(number("PLENA_V2_SRAM_PORT", 1));
        let level = count.trailing_ones();
        assert!(level <= 7, "bounded tree accepts at most 128 terms");
        // Even leaves were written directly to level zero. Each odd merge
        // reads BOTH operands from existing SRAM and stores its result there.
        // No host-held partial is permitted to bypass a modeled SRAM edge.
        for depth in 0..level {
            let mut leaf = self.tree_values(7).await;
            ready = clock.vector(ready, false, port);
            let partial = self.tree_values(depth).await;
            ready = clock.vector(ready, false, port);
            for (x, y) in leaf.iter_mut().zip(partial) {
                *x = rn(*x + y);
            }
            // Independent existing Vector adder. Conservative: fold stages
            // are dependent and do not overlap their SRAM operands.
            ready = clock.existing_arithmetic(ready, *crate::runtime_config::VECTOR_ADD_CYCLES);
            let target = if depth + 1 == level { level } else { 7 };
            self.tree_write(target, 0, &leaf).await;
            ready = clock.vector(ready, true, port);
        }
        ready
    }

    pub(super) async fn execute_v2(&mut self, args: LTileExecArgs, pc: usize) {
        assert_eq!(
            crate::timing::timing_mode(),
            crate::timing::TimingMode::Serial,
            "v2 resource calendar requires serial instruction retirement"
        );
        assert_eq!(args.source_axis, LTileAxis::Row);
        assert_eq!(args.scale_axis, LTileAxis::Row);
        let dst = self.resolve_matrix_view(Some(0), pc).unwrap();
        let src = self.resolve_matrix_view(Some(1), pc).unwrap();
        let coeff = self.resolve_matrix_view(Some(2), pc).unwrap();
        let db = self.reg_file.read_gp(args.destination_register);
        let sb = self.reg_file.read_gp(args.source_register);
        let cb = self.reg_file.read_gp(args.scale_register);
        let native = matches!(
            args.primitive,
            Op::NativeDeltaUpdate | Op::NativeReduceAcc | Op::NativeDecayReduceAcc
        );
        let op = match args.primitive {
            Op::NativeDeltaUpdate => Op::DeltaUpdate,
            Op::NativeReduceAcc => Op::ReduceAcc,
            Op::NativeDecayReduceAcc => Op::DecayReduceAcc,
            x => x,
        };
        if native {
            assert_eq!(self.m_machine.mram.banks(), 64);
            assert_eq!(self.m_machine.mram.bank_width(), 32);
            assert_eq!(self.m_machine.mram.depth_rows(), 256);
            assert_eq!(
                self.m_machine.mram.ty(),
                quantize::MxDataType::Plain(quantize::DataType::Fp(quantize::FpType::BF16))
            );
            if op == Op::DeltaUpdate {
                assert_eq!(
                    src.shape.rows, 1,
                    "native UPDATE requires invariant outer operand"
                );
            }
        }
        let config = self.v_machine.update_lane.config;
        let lanes = config.lanes as usize;
        let width = dst.shape.cols as usize;
        let heads = dst.shape.tile_count as usize;
        let active = width * heads;
        assert!(active <= 2048 && width <= lanes && lanes.is_multiple_of(width));
        let group = lanes / width;
        let is_tree = tree();
        let mut clock = Calendar::default();
        let port = u64::from(number("PLENA_V2_SRAM_PORT", 1));
        let ctx_port = u64::from(number("PLENA_V2_CONTEXT_PORT", 1));
        let dot_latency = number("PLENA_V2_DOT_LATENCY", 6);
        let dot_ii = number("PLENA_V2_DOT_II", 2);

        if op == Op::ReduceBegin {
            assert!(!self.v2.active, "context must be consumed before reset");
            assert_eq!(dst.shape.rows, 1);
            self.v2 = ReductionState {
                active: true,
                values: active,
                rows: 0,
                sums: if is_tree { Vec::new() } else { vec![0.0; 2048] },
            };
            // Tree occupancy is reset as control bits; stale leaves are never
            // read. Context clearing uses the actual L-wide write interface.
            if !is_tree {
                for _ in 0..active.div_ceil(lanes) {
                    clock.context(0, ctx_port, true);
                }
            }
            clock.retire().await;
            return;
        }

        let reducing = matches!(op, Op::ReduceAcc | Op::DecayReduceAcc);
        let finishing = matches!(op, Op::ReduceWrite | Op::ResidualWrite);
        if reducing || finishing {
            assert!(
                self.v2.active && self.v2.values == active,
                "reduction shape/lifetime mismatch"
            );
        }
        if finishing {
            assert_eq!(dst.shape.rows, 1);
            assert!(self.v2.rows.is_power_of_two() && self.v2.rows <= 128);
            let tree_row = self.v2.rows.trailing_zeros();
            let (scalars, scalar_ready) = if op == Op::ResidualWrite {
                self.v2_read(cb, coeff, &[(0, 0)], 0, &mut clock).await
            } else {
                (Vec::new(), 0)
            };
            for first in (0..heads).step_by(group) {
                let last = (first + group).min(heads);
                let lines = (first..last).map(|h| (h as u32, 0)).collect::<Vec<_>>();
                let mut ready = scalar_ready;
                let prediction = if is_tree {
                    // SRAM is read for each supply subchunk. A full-row host
                    // decode is sliced immediately; no P-wide holding buffer
                    // or FP32 context is credited to the tree configuration.
                    let row = self.tree_values(tree_row).await;
                    ready = clock.vector(ready, false, port);
                    row[first * width..last * width].to_vec()
                } else {
                    Vec::new()
                };
                let input = if op == Op::ResidualWrite {
                    let (v, t) = self.v2_read(sb, src, &lines, ready, &mut clock).await;
                    ready = t;
                    v
                } else {
                    Vec::new()
                };
                if !is_tree {
                    ready = clock.context(ready, ctx_port, false);
                }
                let mut result = Vec::with_capacity((last - first) * width);
                for i in first * width..last * width {
                    let pred = if is_tree {
                        prediction[i - first * width]
                    } else {
                        self.v2.sums[i]
                    };
                    let value = if op == Op::ResidualWrite {
                        let beta = scalars[2 * (i / width)];
                        let difference = input[i - first * width] - pred;
                        rn(beta * difference)
                    } else {
                        rn(pred)
                    };
                    result.push(value);
                }
                let done = clock.compute(
                    ready,
                    0,
                    if op == Op::ResidualWrite { 4 } else { 1 },
                    dot_ii,
                );
                self.v2_write(db, dst, &lines, &result, done, &mut clock)
                    .await;
            }
            self.v2.active = false;
            clock.retire().await;
            return;
        }

        assert!(reducing || op == Op::DeltaUpdate);
        let rows = if reducing {
            src.shape.rows
        } else {
            dst.shape.rows
        };
        assert_eq!(src.shape.cols as usize, width);
        assert_eq!(src.shape.tile_count as usize, heads);
        assert!(coeff.shape.rows == rows || coeff.shape.rows == 1);
        if reducing {
            assert!(self.v2.rows + rows <= 128);
        }
        let mut feedback = vec![0u64; active.div_ceil(lanes)];
        let mut row_ready = 0;
        let latency = if op == Op::DeltaUpdate {
            config.latency
        } else if is_tree {
            if op == Op::DecayReduceAcc {
                config.latency + 3
            } else {
                3
            }
        } else {
            dot_latency
                + if op == Op::DecayReduceAcc {
                    config.latency
                } else {
                    0
                }
        };
        let interval = if op == Op::DeltaUpdate {
            config.interval
        } else {
            dot_ii
        };
        // A fixed ring represents pipeline register credits, not a runtime
        // scheduling queue. Backpressure holds the input and does not advance
        // RNG. At most ceil(latency/II) subchunks can be in flight.
        let mut credits = vec![
            0u64;
            if native {
                number("PLENA_NATIVE_RESULT_SLOTS", 4) as usize
            } else {
                latency.div_ceil(interval) as usize
            }
        ];
        let mut launched = 0usize;
        let mut supply_ready = 0;
        let read_width = if native {
            number("PLENA_NATIVE_READ_WIDTH", 512) as usize
        } else {
            lanes
        };
        assert!(read_width >= lanes && read_width <= 2048 && read_width.is_multiple_of(lanes));
        let wide_group = read_width / width;
        let mut sector = None;
        // x/residual is invariant throughout UPDATE. Explicit 4 KiB operand
        // latch, loaded through the real shared SRAM port, not a free input.
        let (outer_latch, outer_ready) = if native && op == Op::DeltaUpdate {
            let lines: Vec<_> = (0..heads).map(|h| (h as u32, 0)).collect();
            self.v2_read(sb, src, &lines, 0, &mut clock).await
        } else {
            (Vec::new(), 0)
        };
        for row in 0..rows {
            if native && row % 32 == 0 {
                let fields: &[usize] = if op == Op::ReduceAcc { &[2] } else { &[0, 1] };
                sector = Some(
                    self.native_sector(
                        fields,
                        row,
                        rows,
                        heads as u32,
                        row_ready.max(supply_ready),
                        &mut clock,
                    )
                    .await,
                );
            }
            let mut input_slots = std::collections::VecDeque::new();
            let mut next_input = 0;
            // Compact scalars occupy <=128 B, held for this logical row only.
            let (compact, compact_ready) = if !native && coeff.broadcast_minor() {
                self.v2_read(
                    cb,
                    coeff,
                    &[(0, if coeff.shape.rows == 1 { 0 } else { row })],
                    row_ready.max(supply_ready),
                    &mut clock,
                )
                .await
            } else {
                (Vec::new(), row_ready)
            };
            let mut leaves_ready = row_ready;
            for first in (0..heads).step_by(group) {
                let last = (first + group).min(heads);
                let chunk = first * width / lanes;
                let slot = launched % credits.len();
                let operands_at = row_ready.max(supply_ready).max(credits[slot]);
                let state_lines = (first..last).map(|h| (h as u32, row)).collect::<Vec<_>>();
                let source_lines = (first..last)
                    .map(|h| (h as u32, if src.shape.rows == 1 { 0 } else { row }))
                    .collect::<Vec<_>>();
                let (state, mut ready) = if native {
                    if first % wide_group == 0 {
                        if first > 0 {
                            input_slots.pop_front();
                        }
                        // Two W-wide slots. Refill only after the previous
                        // slot's last lane group has accepted its operands.
                        while input_slots.len() < 2 && next_input < heads {
                            let lines: Vec<_> = (next_input..(next_input + wide_group).min(heads))
                                .map(|h| (h as u32, row))
                                .collect();
                            let (values, ready) = self
                                .v2_read(
                                    if reducing { sb } else { db },
                                    if reducing { src } else { dst },
                                    &lines,
                                    row_ready.max(supply_ready),
                                    &mut clock,
                                )
                                .await;
                            input_slots.push_back((next_input, values, ready + 1));
                            next_input += wide_group;
                        }
                    }
                    let (wide_first, wide_state, wide_ready) = input_slots.front().unwrap();
                    let start = (first - wide_first) * width;
                    (
                        wide_state[start..start + (last - first) * width].to_vec(),
                        (*wide_ready).max(operands_at),
                    )
                } else {
                    self.v2_read(
                        if reducing { sb } else { db },
                        if reducing { src } else { dst },
                        &state_lines,
                        operands_at,
                        &mut clock,
                    )
                    .await
                };
                let (scalars, scalar_ready) = if native {
                    (Vec::new(), sector.as_ref().unwrap().ready)
                } else if compact.is_empty() {
                    let mut slice = coeff;
                    slice.shape.rows = 1;
                    slice.shape.cols = (2 * (last - first) * width) as u32;
                    slice.shape.tile_count = 1;
                    let stride = coeff.shape.cols.div_ceil(self.m_machine.mram.tile_size());
                    let offset =
                        row * stride * self.m_machine.mram.tile_size() + (2 * first * width) as u32;
                    self.v2_read(cb + offset, slice, &[(0, 0)], operands_at, &mut clock)
                        .await
                } else {
                    (Vec::new(), compact_ready)
                };
                ready = ready.max(scalar_ready);
                let (outer, source_ready) = if native && op == Op::DeltaUpdate {
                    (
                        outer_latch[first * width..last * width].to_vec(),
                        outer_ready,
                    )
                } else if op == Op::DeltaUpdate {
                    self.v2_read(sb, src, &source_lines, operands_at, &mut clock)
                        .await
                } else {
                    (Vec::new(), 0)
                };
                ready = ready.max(source_ready);
                let mut values = Vec::with_capacity(state.len());
                for (i, &s) in state.iter().enumerate() {
                    let h = first + i / width;
                    let (a, b) = if native {
                        let s = sector.as_ref().unwrap();
                        (
                            s.value(0, row, h as u32),
                            if op == Op::ReduceAcc {
                                0.0
                            } else {
                                s.value(1, row, h as u32)
                            },
                        )
                    } else if compact.is_empty() {
                        let local = i / width * 2 * width + i % width;
                        (scalars[local], scalars[local + width])
                    } else {
                        (compact[2 * h], compact[2 * h + 1])
                    };
                    let result = match op {
                        Op::DeltaUpdate => self.v_machine.update_lane.update(i, s, a, b, outer[i]),
                        Op::ReduceAcc => s * a,
                        Op::DecayReduceAcc => {
                            let p = s * a;
                            let decayed = s - p;
                            decayed * b
                        }
                        _ => unreachable!(),
                    };
                    values.push(result);
                }
                if op == Op::DeltaUpdate {
                    let done = clock.compute(ready, 0, config.latency, config.interval);
                    credits[slot] = self
                        .v2_write(db, dst, &state_lines, &values, done, &mut clock)
                        .await;
                    supply_ready = done - u64::from(latency);
                } else if is_tree {
                    for value in &mut values {
                        *value = rn(*value);
                    }
                    let done = clock.compute(
                        ready,
                        0,
                        if op == Op::DecayReduceAcc {
                            config.latency + 3
                        } else {
                            3
                        },
                        dot_ii,
                    );
                    self.tree_write(
                        if self.v2.rows & 1 == 0 { 0 } else { 7 },
                        (first * width) as u32,
                        &values,
                    )
                    .await;
                    credits[slot] = clock.vector(done, true, port);
                    leaves_ready = leaves_ready.max(credits[slot]);
                    supply_ready = done - u64::from(latency);
                } else {
                    // Actual context slot readiness includes both read/write
                    // ports; inactive allocated slots never hide feedback.
                    clock.feedback_wait += feedback[chunk].saturating_sub(ready);
                    let read_done = clock.context(ready.max(feedback[chunk]), ctx_port, false);
                    let done = clock.compute(
                        read_done,
                        feedback[chunk],
                        dot_latency
                            + if op == Op::DecayReduceAcc {
                                config.latency
                            } else {
                                0
                            },
                        dot_ii,
                    );
                    for (i, value) in values.into_iter().enumerate() {
                        self.v2.sums[first * width + i] += value;
                    }
                    let visible = clock.context(done, ctx_port, true);
                    feedback[chunk] = visible;
                    credits[slot] = visible;
                    supply_ready = done - u64::from(latency);
                    clock.end = clock.end.max(visible);
                }
                launched += 1;
            }
            if reducing {
                if is_tree {
                    row_ready = self.tree_fold(self.v2.rows, leaves_ready, &mut clock).await;
                }
                self.v2.rows += 1;
            }
        }
        clock.retire().await;
    }
}
