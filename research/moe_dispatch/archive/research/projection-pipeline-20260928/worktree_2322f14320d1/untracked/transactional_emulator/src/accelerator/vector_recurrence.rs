//! Explicit experimental fused Vector modes. Every data access is to Vector
//! SRAM; a 512-KiB state group must be streamed in <=32-row chunks. No Matrix
//! view, private state cache or FP32 dot accumulator is used here.
use super::Accelerator;
use crate::{ltile_v2::round_bf16, v2_timing::Calendar};
use quantize::{QuantTensor, tensor_from_f32_slice, tensor_to_f32_vec};
use std::collections::BTreeMap;

const W: usize = 2048;
const LEAF: u32 = 5;
const TREE: u32 = 8;
const OUTPUT: u32 = 4;

#[derive(Clone, Copy)]
struct Config {
    rows: usize,
    width: usize,
    level: u32,
    origin: usize,
}
#[derive(Clone, Copy)]
struct Coefficient {
    base: u32,
    hs: u32,
    rs: u32,
    repeat: u32,
    heads: u32,
    extent: u32,
}
impl Coefficient {
    fn unpack(w: u64) -> Self {
        assert_eq!(w >> 62, 0, "reserved Vector coefficient bits");
        let c = Self {
            base: (w & 0x7ffff) as u32,
            hs: ((w >> 19) & 8191) as u32,
            rs: ((w >> 32) & 511) as u32,
            repeat: ((w >> 41) & 7) as u32,
            heads: ((w >> 44) & 63) as u32 + 1,
            extent: ((w >> 50) & 4095) as u32 + 1,
        };
        assert!(
            c.base + c.extent <= 64 * W as u32,
            "Vector coefficient exceeds 256-KiB capacity"
        );
        c
    }
    fn address(self, row: usize, lane: usize, cfg: Config) -> u32 {
        let head = lane / cfg.width;
        assert_eq!(self.heads as usize, W / cfg.width);
        let offset = if cfg.level == 1 {
            lane as u32
        } else {
            ((head as u32) >> self.repeat) * self.hs + row as u32 * self.rs
        };
        assert!(offset < self.extent, "Vector coefficient extent");
        self.base + offset
    }
}
#[derive(Default)]
pub(super) struct State {
    config: Option<Config>,
    coefficients: [Option<Coefficient>; 3],
    invariant: Option<Vec<f32>>,
    tree_rows: usize,
}
impl State {
    pub fn configure(&mut self, shape: u32, control: u32) {
        assert_eq!(shape >> 6, 0, "reserved Vector shape bits");
        assert_eq!(control >> 10, 0, "reserved Vector control bits");
        let c = Config {
            rows: (shape as usize & 31) + 1,
            width: if shape & 32 != 0 { 128 } else { 64 },
            level: control & 7,
            origin: (control >> 3) as usize,
        };
        assert!((1..=4).contains(&c.level) && c.origin + c.rows <= 128);
        assert!(
            c.level == 4 || c.rows == 1,
            "only FSM level has a multirow bound"
        );
        self.config = Some(c);
    }
    pub fn coefficient(&mut self, slot: usize, word: u64) {
        assert!(slot < 3);
        self.coefficients[slot] = Some(Coefficient::unpack(word));
    }
    fn validate_leaf_rows(&self, op: u8) {
        let cfg = self.config.expect("Vector recurrence requires CFG");
        if cfg.level == 4 && matches!(op, 1 | 2) {
            assert!(
                self.tree_rows + cfg.rows <= 128,
                "Vector FSM exceeds 128 tree leaves"
            );
        }
    }
}
fn rn(x: f32) -> f32 {
    f32::from_bits(u32::from(round_bf16(x, None)) << 16)
}
fn shared() -> bool {
    match std::env::var("PLENA_VECTOR_REC_ALU").as_deref() {
        Ok("shared") | Err(_) => true,
        Ok("dedicated") => false,
        _ => panic!("PLENA_VECTOR_REC_ALU must be shared or dedicated"),
    }
}
fn alu_latency() -> u32 {
    let latency =
        std::env::var("PLENA_VECTOR_REC_ALU_LATENCY").map_or(2, |s| s.parse::<u32>().unwrap());
    assert!(latency > 0);
    latency
}
fn shared_timing(clock: &mut Calendar, ready: u64, op: u8) -> u64 {
    let latency = alu_latency();
    match op {
        0 => {
            let p = clock.shared_alu(ready, true, latency, 1);
            let q = clock.shared_alu(ready, true, latency, 1);
            let u = clock.shared_alu(p, false, latency, 1);
            clock.shared_alu(u.max(q), false, latency, 1)
        }
        1 => {
            let p = clock.shared_alu(ready, true, latency, 1);
            let u = clock.shared_alu(p, false, latency, 1);
            clock.shared_alu(u, true, latency, 1)
        }
        2 | 8 => clock.shared_alu(ready, true, latency, 1),
        3 => {
            let d = clock.shared_alu(ready, false, latency, 1);
            clock.shared_alu(d, true, latency, 1)
        }
        9 => {
            let p = clock.shared_alu(ready, true, latency, 1);
            clock.shared_alu(p, false, latency, 1)
        }
        _ => unreachable!(),
    }
}
impl Accelerator {
    async fn vr_read(&mut self, address: u32, ready: u64, clock: &mut Calendar) -> (Vec<f32>, u64) {
        assert_eq!(address % W as u32, 0);
        assert!(address + W as u32 <= 64 * W as u32);
        let q = self.v_machine.vram.read(address).await;
        let port = std::env::var("PLENA_V2_SRAM_PORT").map_or(1, |s| s.parse::<u64>().unwrap());
        (
            tensor_to_f32_vec(q.as_tensor()),
            clock.vector(ready, false, port),
        )
    }
    async fn vr_write(
        &mut self,
        address: u32,
        first: usize,
        values: &[f32],
        ready: u64,
        clock: &mut Calendar,
    ) -> u64 {
        assert_eq!(address % W as u32, 0);
        assert!(address + W as u32 <= 64 * W as u32 && first + values.len() <= W);
        let q = QuantTensor::quantize(tensor_from_f32_slice(values), self.v_machine.vram.ty());
        self.v_machine
            .vram
            .write_bank_words(address, first as u32, q)
            .await;
        let port = std::env::var("PLENA_V2_SRAM_PORT").map_or(1, |s| s.parse::<u64>().unwrap());
        clock.vector(ready, true, port)
    }
    async fn vr_coeff(
        &mut self,
        fields: &[usize],
        row: usize,
        first: usize,
        len: usize,
        ready: u64,
        clock: &mut Calendar,
    ) -> (Vec<Vec<f32>>, u64) {
        let cfg = self.vector_recurrence.config.unwrap();
        let views: Vec<_> = fields
            .iter()
            .map(|&f| self.vector_recurrence.coefficients[f].expect("Vector EXEC without CCFG"))
            .collect();
        let mut addresses = Vec::new();
        let mut snapshots = BTreeMap::new();
        let mut ready = ready;
        for c in &views {
            let list: Vec<_> = (first..first + len)
                .map(|i| c.address(row, i, cfg))
                .collect();
            for &a in &list {
                let base = a / W as u32 * W as u32;
                if let std::collections::btree_map::Entry::Vacant(entry) = snapshots.entry(base) {
                    let (v, t) = self.vr_read(base, ready, clock).await;
                    entry.insert(v);
                    ready = t;
                }
            }
            addresses.push(list);
        }
        // Full-row host decoding is not a free retained 4-KiB operand. Only
        // selected L-wide values survive; each next chunk re-reads the port.
        let values = addresses
            .into_iter()
            .map(|a| {
                a.into_iter()
                    .map(|p| snapshots[&(p / W as u32 * W as u32)][p as usize % W])
                    .collect()
            })
            .collect();
        if cfg.level >= 2 {
            ready += 1;
            clock.end = clock.end.max(ready);
        }
        (values, ready)
    }
    async fn vr_fold(&mut self, count: usize, ready: u64, clock: &mut Calendar) -> u64 {
        let merges = count.trailing_ones();
        let mut ready = ready;
        for depth in 0..merges {
            let (mut leaf, t) = self.vr_read(LEAF * W as u32, ready, clock).await;
            ready = t;
            let (part, t) = self.vr_read((TREE + depth) * W as u32, ready, clock).await;
            ready = t;
            for (x, y) in leaf.iter_mut().zip(part) {
                *x = rn(*x + y);
            }
            ready = clock.existing_arithmetic(ready, *crate::runtime_config::VECTOR_ADD_CYCLES);
            let target = if depth + 1 == merges {
                if count == 127 { OUTPUT } else { TREE + merges }
            } else {
                LEAF
            };
            ready = self
                .vr_write(target * W as u32, 0, &leaf, ready, clock)
                .await;
        }
        ready
    }
    pub(super) async fn execute_vector_recurrence(&mut self, op: u8, db: u32, sb: u32, xb: u32) {
        assert_eq!(
            crate::timing::timing_mode(),
            crate::timing::TimingMode::Serial,
            "Vector recurrence is serial-only"
        );
        let cfg = self
            .vector_recurrence
            .config
            .expect("Vector recurrence requires CFG");
        // Reject the whole instruction before touching a leaf or state row.
        self.vector_recurrence.validate_leaf_rows(op);
        let lanes = self.v_machine.update_lane.config.lanes as usize;
        assert!(cfg.width <= lanes && lanes.is_multiple_of(cfg.width));
        let mut clock = Calendar::default();
        if op == 4 {
            assert!(cfg.level >= 3);
            let (v, _) = self.vr_read(sb, 0, &mut clock).await;
            self.vector_recurrence.invariant = Some(v);
            clock.retire().await;
            return;
        }
        if op == 5 {
            self.vector_recurrence.tree_rows = 0;
            clock.retire().await;
            return;
        }
        if op == 6 {
            let count = self.vector_recurrence.tree_rows;
            assert!(count < 128);
            self.vr_fold(count, 0, &mut clock).await;
            self.vector_recurrence.tree_rows += 1;
            clock.retire().await;
            return;
        }
        if op == 7 {
            assert_eq!(self.vector_recurrence.tree_rows, 128);
            if db != OUTPUT * W as u32 {
                let (v, t) = self.vr_read(OUTPUT * W as u32, 0, &mut clock).await;
                self.vr_write(db, 0, &v, t, &mut clock).await;
            }
            clock.retire().await;
            return;
        }
        assert!(matches!(op, 0..=3 | 8 | 9));
        let rows = if cfg.level == 4 && matches!(op, 0..=2) {
            cfg.rows
        } else {
            1
        };
        let mut ready = 0;
        let mut credits = [0_u64; 4];
        for r in 0..rows {
            let dest = if matches!(op, 1 | 2) && cfg.level == 4 {
                if self.vector_recurrence.tree_rows.is_multiple_of(2) {
                    TREE * W as u32
                } else {
                    LEAF * W as u32
                }
            } else {
                db + r as u32 * W as u32
            };
            let source = sb + r as u32 * W as u32;
            for (first_no, first) in (0..W).step_by(lanes).enumerate() {
                let len = lanes.min(W - first);
                let (mut state, t) = self.vr_read(source, ready, &mut clock).await;
                ready = t;
                state = state[first..first + len].to_vec();
                let fields = match op {
                    0 | 1 => vec![0, 1],
                    2 => vec![2],
                    3 | 8 | 9 => vec![0],
                    _ => unreachable!(),
                };
                let (coeff, t) = self
                    .vr_coeff(&fields, cfg.origin + r, first, len, ready, &mut clock)
                    .await;
                ready = t;
                let input = if op == 0 {
                    if cfg.level >= 3 {
                        self.vector_recurrence
                            .invariant
                            .as_ref()
                            .expect("Vector invariant requires HOLD")[first..first + len]
                            .to_vec()
                    } else {
                        let (v, t) = self.vr_read(xb, ready, &mut clock).await;
                        ready = t;
                        v[first..first + len].to_vec()
                    }
                } else if op == 3 || op == 9 {
                    let (v, t) = self
                        .vr_read(if op == 9 { db } else { xb }, ready, &mut clock)
                        .await;
                    ready = t;
                    v[first..first + len].to_vec()
                } else {
                    Vec::new()
                };
                let mut result = Vec::with_capacity(len);
                for i in 0..len {
                    let s = state[i];
                    let value = match op {
                        0 => self.v_machine.update_lane.update(
                            (first + i) % lanes,
                            s,
                            coeff[0][i],
                            coeff[1][i],
                            input[i],
                        ),
                        1 => {
                            let p = coeff[0][i] * s;
                            let d = s - p;
                            rn(d * coeff[1][i])
                        }
                        2 | 8 => rn(s * coeff[0][i]),
                        3 => {
                            let d = s - input[i];
                            rn(d * coeff[0][i])
                        }
                        9 => rn(input[i] + s * coeff[0][i]),
                        _ => unreachable!(),
                    };
                    result.push(value);
                }
                let launch = ready.max(credits[first_no % 4]);
                let done = if shared() {
                    shared_timing(&mut clock, launch, op)
                } else {
                    let latency = match op {
                        0 => 6,
                        1 => 9,
                        2 => 3,
                        3 => 4,
                        8 => 2,
                        9 => 6,
                        _ => unreachable!(),
                    };
                    clock.compute(launch, 0, latency, 2)
                };
                credits[first_no % 4] = self.vr_write(dest, first, &result, done, &mut clock).await;
                // One operand-latch group is held until the last consumer
                // issues. No uncounted second input bundle may overwrite a
                // key/state operand still needed by a later shared operation.
                ready = if shared() {
                    done - u64::from(alu_latency()) + 1
                } else {
                    launch + 1
                };
            }
            if cfg.level == 4 && matches!(op, 1 | 2) {
                let count = self.vector_recurrence.tree_rows;
                ready = self.vr_fold(count, clock.end, &mut clock).await;
                self.vector_recurrence.tree_rows += 1;
            } else {
                ready = clock.end;
            }
        }
        clock.retire().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_rejects_capacity_and_reserved_fields() {
        let mut s = State::default();
        s.configure(31, 4 | 96 << 3);
        assert_eq!(s.config.unwrap().rows, 32);
        assert!(
            std::panic::catch_unwind(|| {
                let mut s = State::default();
                s.configure(31, 4 | 97 << 3);
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                let mut s = State::default();
                s.coefficient(0, 131070 | (31 << 44) | (3 << 50));
            })
            .is_err()
        );
    }
    #[test]
    fn fsm_rejects_excess_leaves_before_execution() {
        let mut s = State::default();
        s.configure(31 | 32, 4);
        s.tree_rows = 96;
        s.validate_leaf_rows(1);
        s.validate_leaf_rows(2);
        s.tree_rows = 97;
        for op in [1, 2] {
            assert!(std::panic::catch_unwind(|| s.validate_leaf_rows(op)).is_err());
        }
        s.validate_leaf_rows(0); // A state update does not consume tree leaves.
    }
    #[test]
    fn shared_unit_schedule_obeys_dependencies() {
        let mut c = Calendar::default();
        assert_eq!(shared_timing(&mut c, 0, 0), 6);
        assert_eq!(c.launches, 4);
        let second = shared_timing(&mut c, 0, 0);
        assert!(second > 6);
    }
}
