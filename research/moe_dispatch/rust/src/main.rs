#![recursion_limit = "256"]
//! Independent candidate analytical model. NOT native Ramulator or RTL timing.
//! All times are 1 ns cycles. Numeric SRAM validation is a separate executable.
// Simulator-host optimization only: avoid constructing discarded JSON events.
// The condition is evaluated before the event expression; modeled timing and
// recorded trace content remain unchanged.
macro_rules! trace {
    ($sim:expr, $event:expr $(,)?) => {
        if $sim.cfg.record_trace {
            $sim.log($event);
        }
    };
}

mod compute;
mod joint;
mod legacy_chunks;
#[allow(dead_code)]
mod plan;
mod profile_v3;
mod runtime;
mod surplus;
mod v3;
use plan::{Issue, Projection, build_groups};
use runtime::{DmaRequest, NextTask};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};

fn ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}
fn align(a: usize, b: usize) -> usize {
    ceil(a, b) * b
}

#[derive(Clone, Deserialize, Serialize)]
struct Expert {
    id: i64,
    is_shared: bool,
    #[serde(rename = "Me")]
    m: usize,
    #[serde(rename = "H")]
    h: usize,
    #[serde(rename = "F")]
    f: usize,
    #[serde(default)]
    token_indices: Vec<usize>,
    #[serde(default)]
    weights: Value,
}
#[derive(Clone, Deserialize, Serialize)]
struct Workload {
    id: String,
    batch: usize,
    hidden: usize,
    top_k: usize,
    experts: Vec<Expert>,
    #[serde(default)]
    engine_layout: Value,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default)]
struct Config {
    arch: String,
    diagnostic_profile: bool,
    profile_issue_limit: usize,
    weight_bytes_scale: f64,
    lanes: Vec<usize>,
    group: usize,
    dispatch: String,
    fixed_assignment: Vec<usize>,
    tail_partition: bool,
    joint_late_bind: bool,
    joint_feedback: bool,
    joint_pairing: bool,
    joint_age_limit: u8,
    joint_margin_cycles: u64,
    split: String,
    hbm_bytes_per_ns: usize,
    hbm_latency_ns: u64,
    credits: usize,
    diagnostic_credit_expansion: bool,
    arbiter: String,
    ideal_hbm: bool,
    ideal_onchip: bool,
    control_cost: bool,
    window: usize,
    onchip_bytes_per_ns: usize,
    vector_elements_per_ns: usize,
    dot_tail_ns: u64,
    record_trace: bool,
    runtime_fsm: bool,
    next_prefetch: bool,
    next_prefetch_ready_threshold: Option<usize>,
    late_bind_cycles: Option<u64>,
    shared_large: bool,
    surplus_rules: u8,
    surplus_margin_cycles: u64,
    dma_ready_after: u64,
    dma_ready_period: u64,
    dma_ready_cycles: u64,
    max_cycles: u64,
    no_progress_cycles: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            arch: "joint_v1".into(),
            diagnostic_profile: false,
            profile_issue_limit: 4096,
            weight_bytes_scale: 1.0,
            lanes: vec![4, 2],
            group: 4,
            dispatch: "dynamic".into(),
            fixed_assignment: vec![],
            tail_partition: false,
            joint_late_bind: true,
            joint_feedback: true,
            joint_pairing: true,
            joint_age_limit: 8,
            joint_margin_cycles: 64,
            split: "none".into(),
            hbm_bytes_per_ns: 256,
            hbm_latency_ns: 64,
            credits: 256,
            diagnostic_credit_expansion: false,
            arbiter: "stock".into(),
            ideal_hbm: false,
            ideal_onchip: false,
            control_cost: true,
            window: 8,
            onchip_bytes_per_ns: 384,
            vector_elements_per_ns: 32,
            dot_tail_ns: 20,
            record_trace: false,
            runtime_fsm: true,
            next_prefetch: true,
            next_prefetch_ready_threshold: None,
            late_bind_cycles: None,
            shared_large: false,
            surplus_rules: 0,
            surplus_margin_cycles: 64,
            dma_ready_after: 0,
            dma_ready_period: 1,
            dma_ready_cycles: 1,
            max_cycles: 200_000_000,
            no_progress_cycles: 1_000_000,
        }
    }
}

#[derive(Clone)]
struct Banks {
    free: Vec<u64>,
    words: u64,
    wait: u64,
}
impl Banks {
    fn new(n: usize) -> Self {
        Self {
            free: vec![0; n],
            words: 0,
            wait: 0,
        }
    }
    fn access(&mut self, t: u64, spans: &[(usize, usize)], read: bool, rmw: bool) -> u64 {
        let mut counts = vec![0u64; self.free.len()];
        for &(a, b) in spans {
            if b > 0 {
                for w in a / 16..ceil(a + b, 16) {
                    counts[w % self.free.len()] += 1;
                }
            }
        }
        let mut end = t;
        for (i, n) in counts.into_iter().enumerate() {
            if n == 0 {
                continue;
            }
            let start = t.max(self.free[i]);
            self.wait += start - t;
            self.words += n * if rmw { 2 } else { 1 };
            self.free[i] = start + n * if rmw { 5 } else { 1 };
            end = end.max(self.free[i] + if read && !rmw { 1 } else { 0 });
        }
        end
    }
}
#[derive(Default, Serialize)]
struct Stats {
    useful_macs: u64,
    issued_macs: u64,
    issues: u64,
    weight_bytes: u64,
    x_stage_bytes: u64,
    copy_bytes: u64,
    remote_copy_bytes: u64,
    z_exchange_bytes: u64,
    accumulator_rmw_bytes: u64,
    control_cycles: u64,
    vector_service_cycles: u64,
    weight_peak_bytes: usize,
    x_peak_bytes: usize,
    workspace_peak_bytes: usize,
    contexts_peak: usize,
    arithmetic_active_cycles: u64,
    done_cycle: u64,
    prediction_absolute_error_cycles: u64,
    prediction_actual_cycles: u64,
    front_states: BTreeMap<String, u64>,
    expert_completions: usize,
    next_bindings: usize,
    next_prefetch_tiles: usize,
    next_ready_at_promotion: usize,
    next_inflight_at_promotion: usize,
    next_weight_wait_cycles: u64,
    next_peak_bytes: usize,
    dma_accepted: u64,
    dma_landed: u64,
    dma_backpressure_cycles: u64,
    eligible_wait_max_cycles: u64,
}
#[derive(Clone)]
struct Tile {
    n_start: usize,
    k_start: usize,
    nv: usize,
    kv: usize,
    slot: Option<usize>,
    release: u64,
    sent: usize,
    acks: usize,
    bytes: usize,
    ready: bool,
    retired: bool,
}
#[derive(Clone)]
struct Micro {
    spec: Issue,
    tile: usize,
    chunk: usize,
}
struct Run {
    tiles: Vec<Tile>,
    issues: Vec<Micro>,
    pos: usize,
    admit: usize,
    send_cursor: usize,
    chunk: usize,
    converted: bool,
    converting: bool,
    committed: BTreeMap<(usize, usize), usize>,
}
impl Run {
    fn new(m: usize, n: usize, k: usize, n_start: usize, mc: usize, g: usize) -> Self {
        let groups = build_groups(Projection { m, n, k, n_start }, mc, g).unwrap();
        let mut tiles = vec![];
        let mut issues = vec![];
        for group in groups {
            let chunk = (group.n_start - n_start) / (g * 4);
            let first = tiles.len();
            for t in group.tiles {
                tiles.push(Tile {
                    n_start: t.n_start,
                    k_start: t.k_start,
                    nv: t.n_valid,
                    kv: t.k_valid,
                    slot: None,
                    release: 0,
                    sent: 0,
                    acks: 0,
                    bytes: t.native_bytes as usize,
                    ready: false,
                    retired: false,
                });
            }
            for i in group.issues {
                let idx = first + (i.n_start - group.n_start) / 4;
                issues.push(Micro {
                    spec: i,
                    tile: idx,
                    chunk,
                });
            }
        }
        Self {
            tiles,
            issues,
            pos: 0,
            admit: 0,
            send_cursor: 0,
            chunk: 0,
            converted: false,
            converting: false,
            committed: BTreeMap::new(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct XTag {
    k: usize,
    m: usize,
}
struct XSlot {
    tag: Option<XTag>,
    ready: u64,
    busy_until: u64,
}
#[derive(Clone)]
struct Session {
    e: usize,
    split: bool,
    f0: usize,
    fn_: usize,
    h0: usize,
    hn: usize,
    phase: usize,
    start: u64,
    predicted_duration: u64,
    issues_at_start: u64,
}
struct Core {
    m: usize,
    wslots: usize,
    wb: Banks,
    xb: Banks,
    ab: Banks,
    session: Option<Session>,
    next: Option<NextTask>,
    incoming: Option<Tile>,
    ahead: Option<surplus::PhaseAhead>,
    surplus_thresholds: Option<surplus::Thresholds>,
    prefetch_promoted_at: Option<u64>,
    dma_wait_since: Option<u64>,
    run: Option<Run>,
    xs: Vec<XSlot>,
    free_slots: Vec<usize>,
    live_tiles: Vec<Option<usize>>,
    contexts: usize,
    feed_until: u64,
    arithmetic_until: u64,
    blocked_until: u64,
    weight_cache: Option<usize>,
    reserved: usize,
    capacity: usize,
    stats: Stats,
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
enum Event {
    RuntimeReturn(DmaRequest),
    RuntimeAck(DmaRequest),
    Return(usize, usize, usize),
    Ack(usize, usize),
    Feed(usize, usize, usize, bool),
    Dot(usize, usize),
    Commit(usize, usize),
    Begin(usize, usize),
    ChunkConverted(usize),
    VectorDone(usize),
    DrainDone(usize),
    Feedback(usize, usize, u64),
    ZCopyDone(usize, usize, usize, usize, usize),
}

struct Sim {
    // Host/kernel clock origin for serial outer token chunks. The original
    // single invocation keeps zero; the physical DMA ready pattern never resets.
    time_origin: u64,
    w: Workload,
    cfg: Config,
    cores: Vec<Core>,
    now: u64,
    seq: u64,
    events: BinaryHeap<Reverse<(u64, u64, Event)>>,
    credit_used: usize,
    credit_peak: usize,
    control_free: u64,
    bus_free: u64,
    vector_free: u64,
    rr_desc: usize,
    rr_hbm: usize,
    status: Vec<u8>,
    fixed: Vec<usize>,
    split_ready: Vec<Vec<bool>>,
    split_drained: Vec<Vec<bool>>,
    trace: Vec<Value>,
    decisions: u64,
    deferrals: u64,
    decision_needed: bool,
    decision_after: u64,
    decision_cursor: usize,
    all_done_at: Option<u64>,
    combine_cycles: u64,
    unassigned_peak: usize,
    pending: VecDeque<usize>,
    input_cursor: usize,
    pending_dma: Option<DmaRequest>,
    outstanding_dma: BTreeMap<u64, DmaRequest>,
    dma_serial: u64,
    last_progress: u64,
    runtime_decision_paid: bool,
    runtime_candidate_mask: u8,
    late_bind_wait_cycles: u64,
    prefetch_gate_wait: [u64; 2], // Observer: unsent Current / enough ready tiles.
    promotion_pending: Vec<Option<u64>>,
    input_backpressure_cycles: u64,
    dispatch_audit: Vec<Value>, // Observer history, not a hardware task queue.
    diagnostics: surplus::Diagnostics,
    feedback_q8: Vec<[u64; 4]>,
    feedback_updates: u64,
    tail_decision_ready: Option<u64>,
    tail_partition_count: u64,
    joint: joint::JointState,
    profile_v3: profile_v3::Profile,
}

impl Sim {
    fn dma_ready_now(&self) -> bool {
        let absolute = self.time_origin + self.now;
        absolute >= self.cfg.dma_ready_after
            && absolute % self.cfg.dma_ready_period < self.cfg.dma_ready_cycles
    }
    fn event(&mut self, t: u64, e: Event) {
        assert!(t >= self.now);
        self.seq += 1;
        self.events.push(Reverse((t, self.seq, e)));
    }
    fn log(&mut self, v: Value) {
        if self.cfg.record_trace {
            self.trace.push(v);
        }
    }
    fn layout(&self, e: usize, c: usize, split: bool) -> Option<&Value> {
        if self.w.engine_layout.is_null() {
            None
        } else {
            Some(&self.w.engine_layout[if split { "paired_n" } else { "whole" }][e][c])
        }
    }
    fn allocation(&self, c: usize, name: &str) -> (usize, usize) {
        let s = self.cores[c].session.as_ref().unwrap();
        if let Some(layout) = self.layout(s.e, c, s.split) {
            let a = &layout["allocations"][name];
            return (
                a["base"].as_u64().expect("compiler allocation base") as usize,
                a["stride"].as_u64().expect("compiler stride") as usize,
            );
        }
        let e = &self.w.experts[s.e];
        let mut base = self.cores[c].reserved;
        let x = base;
        base = align(base + 2 * e.m * e.h, 32);
        let z = base;
        base = align(base + 2 * e.m * e.f, 32);
        let up = base;
        base = align(base + 2 * e.m * s.fn_, 32);
        let y = base;
        base = align(base + 4 * e.m * s.hn, 32);
        match name {
            "x" => (x, e.h * 2),
            "z" | "gate_z" | "gate" => (z, e.f * 2),
            "up" => (up, s.fn_ * 2),
            "y" | "producer_y" => (y, s.hn * 4),
            "scratch0" => (base, 32),
            "scratch1" => (base + e.m * self.cfg.group * 32, 32),
            _ => panic!("bad allocation"),
        }
    }
    fn spans(
        base: usize,
        stride: usize,
        rows: usize,
        col: usize,
        width: usize,
    ) -> Vec<(usize, usize)> {
        (0..rows)
            .map(|r| (base + r * stride + col, width))
            .collect()
    }
    fn result_address(&self, c: usize, kind: &str, e: usize) -> (usize, usize) {
        if !self.w.engine_layout.is_null() {
            let r = &self.w.engine_layout["cores"][c]["result_layout"];
            let v = if kind == "inbox" {
                r["inboxes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|x| x["expert_id"].as_i64() == Some(self.w.experts[e].id))
                    .unwrap()
            } else {
                &r[kind]
            };
            let stride = v["row_stride_bytes"]
                .as_u64()
                .map(|x| x as usize)
                .unwrap_or_else(|| v["shape"][1].as_u64().unwrap() as usize * 4);
            return (v["address_bytes"].as_u64().unwrap() as usize, stride);
        }
        let (_, n) = self.bounds(self.w.hidden, c);
        let base = 4096 * self.cores[c].m / 6;
        match kind {
            "original_x" => (base, 2 * n),
            "inbox" => (
                base + 2 * self.w.batch * n
                    + 4 * n * self.w.experts[..e].iter().map(|x| x.m).sum::<usize>(),
                4 * n,
            ),
            _ => (
                base + 2 * self.w.batch * n
                    + 4 * n * self.w.experts.iter().map(|x| x.m).sum::<usize>(),
                4 * n,
            ),
        }
    }
    // Transfers have explicit source read, shared link and destination write.
    // Block phases are conservative; no uncharged streaming overlap is assumed.
    fn transfer_io(
        &mut self,
        src: usize,
        dst: usize,
        t: u64,
        ss: &[(usize, usize)],
        ds: &[(usize, usize)],
        z: bool,
    ) -> u64 {
        let bytes: usize = ss.iter().map(|x| x.1).sum();
        assert_eq!(bytes, ds.iter().map(|x| x.1).sum::<usize>());
        assert!(ss.iter().all(|&(a, b)| a + b <= self.cores[src].capacity));
        assert!(ds.iter().all(|&(a, b)| a + b <= self.cores[dst].capacity));
        if bytes == 0 {
            return t;
        }
        self.cores[dst].stats.copy_bytes += bytes as u64;
        if src != dst {
            self.cores[dst].stats.remote_copy_bytes += bytes as u64;
        }
        if z {
            self.cores[dst].stats.z_exchange_bytes += bytes as u64;
        }
        if self.cfg.ideal_onchip {
            return t;
        }
        let read = self.cores[src].ab.access(t, ss, true, false);
        let start = read.max(self.bus_free);
        self.bus_free = start + ceil(bytes, self.cfg.onchip_bytes_per_ns) as u64;
        self.cores[dst].ab.access(self.bus_free, ds, false, false)
    }
    fn bounds(&self, n: usize, c: usize) -> (usize, usize) {
        let total: usize = self.cfg.lanes.iter().sum();
        let before: usize = self.cfg.lanes[..c].iter().sum();
        let bands = ceil(n, 4);
        let lo = (bands * before / total * 4).min(n);
        let hi = (bands * (before + self.cfg.lanes[c]) / total * 4).min(n);
        (lo, hi - lo)
    }
    fn workspace(&self, e: usize, c: usize, split: bool) -> usize {
        if let Some(l) = self.layout(e, c, split) {
            return l["workspace_bytes"].as_u64().unwrap() as usize;
        }
        let x = &self.w.experts[e];
        let (_, f) = if split { self.bounds(x.f, c) } else { (0, x.f) };
        let (_, h) = if split { self.bounds(x.h, c) } else { (0, x.h) };
        // X, full gate/Z alias (write after last G read), local U, Y, scratch.
        align(2 * x.m * x.h, 64)
            + align(2 * x.m * f, 64)
            + align(2 * x.m * x.f, 64)
            + align(4 * x.m * h, 64)
            + align(2 * x.m * self.cfg.group * 32, 64)
    }
    fn fits(&self, e: usize, c: usize, split: bool) -> bool {
        if let Some(l) = self.layout(e, c, split) {
            return l["feasible"].as_bool().unwrap();
        }
        self.cores[c].reserved + self.workspace(e, c, split) <= self.cores[c].capacity
    }
    fn prediction(&self, e: usize, c: usize, concurrent: usize, split: bool) -> u64 {
        // Service estimate uses dimensions/current queue state only, never event times.
        let x = &self.w.experts[e];
        let (_, f) = if split { self.bounds(x.f, c) } else { (0, x.f) };
        let (_, h) = if split { self.bounds(x.h, c) } else { (0, x.h) };
        let mut work = 0u64;
        let mut weights = 0usize;
        for (n, k) in [(f, x.h), (f, x.h), (h, x.f)] {
            let tiles = ceil(n, 4) * ceil(k, 512);
            let issues = tiles * ceil(x.m, self.cores[c].m);
            weights += 2 * n * k;
            let wfeed = ceil(4096, self.cores[c].wb.free.len() * 16);
            let xfeed = 16usize; // full physical X row-vector service on 4 banks/M.
            work += (issues * (wfeed.max(xfeed) + 1)) as u64;
            work += (ceil(n, 4 * self.cfg.group) * self.cfg.dot_tail_ns as usize) as u64;
        }
        let memory = if self.cfg.ideal_hbm {
            0
        } else {
            let bw = self
                .cfg
                .hbm_bytes_per_ns
                .min((self.cfg.credits * 32) / (self.cfg.hbm_latency_ns.max(1) as usize))
                .max(1);
            ceil(weights * concurrent.max(1), bw) as u64
        };
        let vec = ceil(3 * x.m * f, self.cfg.vector_elements_per_ns) as u64;
        work.max(memory) + vec + 3 * self.cfg.hbm_latency_ns + 64
    }
    fn remaining_estimate(&self, c: usize) -> u64 {
        self.cores[c].session.as_ref().map_or(0, |s| {
            s.predicted_duration
                .saturating_sub(self.now - s.start)
                .max(1)
        })
    }
    fn reserve_start(&mut self, e: usize, c: usize, split: bool) {
        assert!(self.cores[c].session.is_none() && self.fits(e, c, split));
        let x = self.w.experts[e].clone();
        let (f0, fn_) = if split { self.bounds(x.f, c) } else { (0, x.f) };
        let (h0, hn) = if split { self.bounds(x.h, c) } else { (0, x.h) };
        let workspace = self.workspace(e, c, split);
        let pred = self.prediction(e, c, self.cores.len(), split);
        self.cores[c].stats.workspace_peak_bytes = self.cores[c]
            .stats
            .workspace_peak_bytes
            .max(workspace + self.cores[c].reserved);
        self.cores[c].session = Some(Session {
            e,
            split,
            f0,
            fn_,
            h0,
            hn,
            phase: 0,
            start: self.now,
            predicted_duration: pred,
            issues_at_start: self.cores[c].stats.issues,
        });
        self.load_surplus_thresholds(c);
        let mut ready = self.now;
        let (dst, dstride) = self.allocation(c, "x");
        for src in 0..self.cores.len() {
            let (lo, n) = self.bounds(x.h, src);
            let (base, stride) = self.result_address(src, "original_x", e);
            let ss: Vec<_> = (0..x.m)
                .map(|r| {
                    (
                        base + x.token_indices.get(r).copied().unwrap_or(r) * stride,
                        2 * n,
                    )
                })
                .collect();
            let ds = Self::spans(dst, dstride, x.m, lo * 2, n * 2);
            ready = ready.max(self.transfer_io(src, c, self.now, &ss, &ds, false));
        }
        self.cores[c].blocked_until = ready;
        self.event(ready, Event::Begin(c, 0));
        let event = if self.cfg.runtime_fsm {
            "current_start"
        } else {
            "commit_owner"
        };
        trace!(
            self,
            json!({"event":event,"cycle":self.now,"task":e,"expert":x.id,"core":c,"split":split,"workspace":workspace,"predicted_duration":pred})
        );
    }
    fn dispatch(&mut self) {
        if self.cfg.runtime_fsm {
            self.dispatch_runtime();
            return;
        }
        if !self.decision_needed || self.now < self.control_free || self.now < self.decision_after {
            return;
        }
        let idle: Vec<_> = (0..self.cores.len())
            .filter(|&c| self.cores[c].session.is_none())
            .collect();
        let pending: Vec<_> = (0..self.status.len())
            .filter(|&e| self.status[e] == 0)
            .take(self.cfg.window)
            .collect();
        self.unassigned_peak = self.unassigned_peak.max(pending.len());
        if idle.is_empty() || pending.is_empty() {
            self.decision_needed = false;
            return;
        }
        // Descriptor scan/lookup is paid before the next pass makes the selection.
        if self.decision_cursor == 0 && self.cfg.control_cost {
            let p = pending.len();
            let n = self.cores.len();
            // Conservative serial controller: LUT accesses plus bounded action scoring.
            let actions = p * n + p * p * n * n + if self.cfg.split != "none" { p } else { 0 };
            let cost = if self.cfg.dispatch == "fixed" {
                (2 * p + 2) as u64
            } else {
                (2 * p * n + actions * (p * n + 1) + 2) as u64
            };
            self.control_free = self.now + cost;
            self.cores[idle[0]].stats.control_cycles += cost;
            self.decision_cursor = 1;
            return;
        }
        self.decision_cursor = 0;
        self.decisions += 1;
        self.decision_needed = false;
        // Candidate: two whole experts (pair matching), or one permitted split expert.
        let mut actions: Vec<(u64, usize, usize, Option<(usize, usize)>, bool)> = vec![];
        for &e in &pending {
            for &c in &idle {
                if !self.fits(e, c, false) || (self.cfg.dispatch == "fixed" && self.fixed[e] != c) {
                    continue;
                }
                let duration = self.prediction(e, c, self.cores.len(), false);
                if idle.len() == 1 && self.cfg.dispatch != "fixed" {
                    let best_wait = (0..self.cores.len())
                        .filter(|&d| d != c && self.fits(e, d, false))
                        .map(|d| self.remaining_estimate(d) + self.prediction(e, d, 1, false))
                        .min()
                        .unwrap_or(u64::MAX);
                    if best_wait.saturating_add(32) < duration
                        && self.status.iter().filter(|&&s| s == 0).count() <= self.cfg.window
                    {
                        continue;
                    }
                }
                actions.push((duration, e, c, None, false));
                for &d in &idle {
                    if d == c {
                        continue;
                    }
                    for &f in &pending {
                        if f == e
                            || !self.fits(f, d, false)
                            || (self.cfg.dispatch == "fixed" && self.fixed[f] != d)
                        {
                            continue;
                        }
                        actions.push((
                            duration.max(self.prediction(f, d, 2, false)),
                            e,
                            c,
                            Some((f, d)),
                            false,
                        ));
                    }
                }
            }
        }
        if idle.len() == 2 && self.cfg.split != "none" {
            for &e in &pending {
                if (0..2).all(|c| self.fits(e, c, true)) {
                    let mut dur = (0..2)
                        .map(|c| self.prediction(e, c, 2, true))
                        .max()
                        .unwrap();
                    dur += ceil(
                        2 * self.w.experts[e].m * self.w.experts[e].f,
                        self.cfg.onchip_bytes_per_ns,
                    ) as u64;
                    actions.push((dur, e, 0, Some((e, 1)), true));
                }
            }
        }
        if actions.is_empty() {
            if self.cores.iter().all(|c| c.session.is_none()) {
                panic!("no feasible expert placement; refuse uncharged spill");
            }
            self.deferrals += 1;
            return;
        }
        // Compare the SAME pending set for every action, including leaving a core idle.
        // Remaining tasks are forecast by a small list scheduler; no future events read.
        for a in &mut actions {
            let mut load: Vec<u64> = (0..self.cores.len())
                .map(|c| self.remaining_estimate(c))
                .collect();
            let (_, e, c, other, split) = *a;
            load[c] += self.prediction(e, c, self.cores.len(), split);
            if let Some((f, d)) = other {
                load[d] += self.prediction(f, d, self.cores.len(), split);
            }
            if split {
                let mut transfer = 0u64;
                for src in 0..2 {
                    let dst = 1 - src;
                    let (_, nf) = self.bounds(self.w.experts[e].f, src);
                    let b = 2 * self.w.experts[e].m * nf;
                    if !self.cfg.ideal_onchip {
                        transfer += (ceil(b, self.cores[src].ab.free.len() * 16)
                            + ceil(b, self.cfg.onchip_bytes_per_ns)
                            + ceil(b, self.cores[dst].ab.free.len() * 16))
                            as u64;
                    }
                }
                let joined = *load.iter().max().unwrap() + transfer;
                load.fill(joined);
            }
            for &f in &pending {
                if f == e || other.is_some_and(|(x, _)| x == f) {
                    continue;
                }
                if let Some(d) = (0..self.cores.len())
                    .filter(|&d| {
                        self.fits(f, d, false)
                            && (self.cfg.dispatch != "fixed" || self.fixed[f] == d)
                    })
                    .min_by_key(|&d| load[d] + self.prediction(f, d, self.cores.len(), false))
                {
                    load[d] += self.prediction(f, d, self.cores.len(), false);
                }
            }
            a.0 = *load.iter().max().unwrap();
        }
        if self.cfg.split == "forced" && idle.len() == 2 && actions.iter().any(|a| a.4) {
            actions.retain(|a| a.4);
        }
        actions.sort_by_key(|a| (a.0, a.4, a.1, a.2));
        let (_, e, c, other, split) = actions[0];
        self.status[e] = 1;
        self.reserve_start(e, c, split);
        if let Some((f, d)) = other {
            self.status[f] = 1;
            self.reserve_start(f, d, split);
        }
        self.decision_needed = true;
    }
    fn begin(&mut self, c: usize, phase: usize) {
        if phase != 0 {
            self.cores[c].prefetch_promoted_at = None;
        }
        let s = self.cores[c].session.as_mut().unwrap();
        s.phase = phase;
        let ss = s.clone();
        let e = &self.w.experts[ss.e];
        let (n, k, start) = if phase < 2 {
            (ss.fn_, e.h, ss.f0)
        } else {
            (ss.hn, e.f, ss.h0)
        };
        assert_eq!(self.cores[c].contexts, 0);
        let prefetched = if phase == 0 {
            self.cores[c].incoming.take()
        } else {
            self.cores[c].ahead.take().map(|a| {
                assert_eq!(a.phase, phase);
                a.tile
            })
        };
        let next_occupied = self.cores[c]
            .next
            .as_ref()
            .is_some_and(|v| v.first.is_some()) as usize;
        assert_eq!(
            self.cores[c].free_slots.len()
                + next_occupied
                + prefetched.is_some() as usize
                + usize::from(self.cores[c].ahead.is_some()),
            self.cores[c].wslots
        );
        let mut run = Run::new(e.m, n, k, start, self.cores[c].m, self.cfg.group);
        if let Some(tile) = prefetched {
            assert_eq!(
                (tile.n_start, tile.k_start, tile.bytes),
                (
                    run.tiles[0].n_start,
                    run.tiles[0].k_start,
                    run.tiles[0].bytes
                )
            );
            self.cores[c].live_tiles[tile.slot.unwrap()] = Some(0);
            run.send_cursor = usize::from(tile.sent == tile.bytes);
            run.admit = 1;
            run.tiles[0] = tile;
        }
        self.cores[c].run = Some(run);
        self.cores[c].weight_cache = None;
        for x in &mut self.cores[c].xs {
            x.tag = None;
            x.ready = 0;
            x.busy_until = 0;
        }
        self.cores[c].blocked_until = self.now;
        trace!(
            self,
            json!({"event":"projection_start","cycle":self.now,"core":c,"expert":e.id,"phase":phase,"M":e.m,"N":n,"K":k,"N_start":start})
        );
    }
    fn vector_io(
        &mut self,
        c: usize,
        elements: usize,
        ss: &[(usize, usize)],
        ds: &[(usize, usize)],
    ) -> u64 {
        assert!(
            ss.iter()
                .chain(ds.iter())
                .all(|&(a, b)| a + b <= self.cores[c].capacity)
        );
        let read = if self.cfg.ideal_onchip {
            self.now
        } else {
            self.cores[c].ab.access(self.now, ss, true, false)
        };
        let service = if self.cfg.ideal_onchip {
            0
        } else {
            ceil(elements, self.cfg.vector_elements_per_ns) as u64 + 16
        };
        let start = read.max(self.vector_free);
        self.vector_free = start + service;
        self.cores[c].stats.vector_service_cycles += service;
        if self.cfg.ideal_onchip {
            self.vector_free
        } else {
            self.cores[c].ab.access(self.vector_free, ds, false, false)
        }
    }
    fn end_projection(&mut self, c: usize) {
        let s = self.cores[c].session.as_ref().unwrap().clone();
        let e = self.w.experts[s.e].clone();
        trace!(
            self,
            json!({"event":"projection_done","cycle":self.now,"core":c,"expert":e.id,"phase":s.phase})
        );
        self.cores[c].run = None;
        if s.phase == 0 {
            self.event(self.now, Event::Begin(c, 1));
        } else if s.phase == 1 {
            self.cores[c].session.as_mut().unwrap().phase = 2;
            let (g, gs) = self.allocation(c, "z");
            let (u, us) = self.allocation(c, "up");
            let mut ss = Self::spans(g, gs, e.m, s.f0 * 2, s.fn_ * 2);
            ss.extend(Self::spans(u, us, e.m, 0, s.fn_ * 2));
            let ds = Self::spans(g, gs, e.m, s.f0 * 2, s.fn_ * 2);
            let end = self.vector_io(c, e.m * s.fn_, &ss, &ds);
            self.cores[c].blocked_until = end;
            self.event(end, Event::VectorDone(c));
        } else {
            self.cores[c].session.as_mut().unwrap().phase = 5;
            let mut end = self.now;
            for dst in 0..self.cores.len() {
                let (lo, n) = self.bounds(e.h, dst);
                let a = s.h0.max(lo);
                let b = (s.h0 + s.hn).min(lo + n);
                if b > a {
                    let (yb, ys) = self.allocation(c, "y");
                    let (db, ds) = self.result_address(dst, "inbox", s.e);
                    let src = Self::spans(yb, ys, e.m, (a - s.h0) * 4, (b - a) * 4);
                    let target = Self::spans(db, ds, e.m, (a - lo) * 4, (b - a) * 4);
                    end = end.max(self.transfer_io(c, dst, self.now, &src, &target, false));
                }
            }
            self.cores[c].blocked_until = end;
            self.event(end, Event::DrainDone(c));
        }
    }
    fn handle(&mut self, event: Event) {
        self.last_progress = self.now;
        match event {
            Event::RuntimeReturn(req) => self.return_runtime(req),
            Event::RuntimeAck(req) => self.ack_runtime(req),
            Event::Return(c, t, offset) => {
                self.profile_return();
                let tile = &self.cores[c].run.as_ref().unwrap().tiles[t];
                let slot = tile.slot.unwrap();
                let row_stride = align(tile.kv * 2, 32);
                let row = offset / row_stride;
                let col = offset % row_stride;
                assert!(row < tile.nv);
                let dst = slot * 4096 + row * 1024 + col;
                let end = if self.cfg.ideal_onchip {
                    self.now
                } else {
                    self.cores[c]
                        .wb
                        .access(self.now, &[(dst, 32)], false, false)
                };
                self.event(end, Event::Ack(c, t));
            }
            Event::Ack(c, t) => {
                self.profile_landed();
                assert!(self.credit_used > 0);
                self.credit_used -= 1;
                let tile = &mut self.cores[c].run.as_mut().unwrap().tiles[t];
                tile.acks += 32;
                assert!(tile.acks <= tile.bytes);
                if tile.acks == tile.bytes {
                    tile.ready = true;
                }
                if tile.ready {
                    let s = self.cores[c].session.as_ref().unwrap();
                    self.profile_weight_ready(c, s.e, s.phase, t);
                }
            }
            Event::Feed(c, t, x, last) => {
                self.cores[c].xs[x].busy_until = self.now;
                if last {
                    let tile = &mut self.cores[c].run.as_mut().unwrap().tiles[t];
                    tile.retired = true;
                    let freed = tile.slot.take().unwrap();
                    self.cores[c].live_tiles[freed] = None;
                    self.cores[c].free_slots.push(freed);
                    trace!(
                        self,
                        json!({"event":"weight_slot_free","cycle":self.now,"core":c,"slot":freed}),
                    );
                }
            }
            Event::Dot(c, i) => {
                self.profile_dot(c, i);
                let spec = self.cores[c].run.as_ref().unwrap().issues[i].spec.clone();
                let s = self.cores[c].session.as_ref().unwrap();
                let start = if s.phase < 2 { s.f0 } else { s.h0 };
                let local = (spec.n_start - start) % (4 * self.cfg.group);
                let (base, _) = self.allocation(c, "scratch0");
                let spans: Vec<_> = (0..spec.m_valid)
                    .map(|r| {
                        (
                            base + (local / 4) * self.w.experts[s.e].m * 32
                                + (spec.m_start + r) * 32,
                            spec.n_valid * 4,
                        )
                    })
                    .collect();
                let done = if self.cfg.ideal_onchip {
                    self.now + 2
                } else {
                    self.cores[c].ab.access(self.now, &spans, true, true)
                };
                self.cores[c].stats.accumulator_rmw_bytes +=
                    (spec.m_valid * spec.n_valid * 8) as u64;
                self.event(done, Event::Commit(c, i));
            }
            Event::Commit(c, i) => {
                self.profile_commit(c, i);
                let run = self.cores[c].run.as_mut().unwrap();
                let s = &run.issues[i].spec;
                let v = run.committed.entry((s.m_start, s.n_start)).or_default();
                assert_eq!(*v, s.k_start / 512);
                *v += 1;
                if self.cfg.record_trace {
                    let s = self.cores[c].session.as_ref().unwrap();
                    trace!(
                        self,
                        json!({"event":"k_commit","cycle":self.now,"core":c,
                        "task":s.e,"phase":s.phase,"issue":i})
                    );
                }
                assert!(self.cores[c].contexts > 0);
                self.cores[c].contexts -= 1;
            }
            Event::Begin(c, p) => self.begin(c, p),
            Event::ChunkConverted(c) => {
                let run = self.cores[c].run.as_mut().unwrap();
                run.converted = true;
                run.converting = false;
            }
            Event::VectorDone(c) => {
                let s = self.cores[c].session.as_ref().unwrap().clone();
                trace!(
                    self,
                    json!({"event":"activation_done","cycle":self.now,"core":c,"task":s.e})
                );
                if !s.split {
                    self.event(self.now, Event::Begin(c, 4));
                } else {
                    self.split_ready[s.e][c] = true;
                    self.cores[c].session.as_mut().unwrap().phase = 3;
                    if self.split_ready[s.e].iter().all(|x| *x) {
                        let mut done = self.now;
                        for src in 0..2 {
                            let other = 1 - src;
                            let sess = self.cores[src].session.as_ref().unwrap().clone();
                            let m = self.w.experts[s.e].m;
                            let (sb, st) = self.allocation(src, "z");
                            let (db, dt) = self.allocation(other, "z");
                            let a = Self::spans(sb, st, m, sess.f0 * 2, sess.fn_ * 2);
                            let b = Self::spans(db, dt, m, sess.f0 * 2, sess.fn_ * 2);
                            let landed = self.transfer_io(src, other, self.now, &a, &b, true);
                            done = done.max(landed);
                            self.event(
                                landed,
                                Event::ZCopyDone(src, other, s.e, sess.f0, sess.fn_),
                            );
                        }
                        for d in 0..2 {
                            self.cores[d].blocked_until = done;
                            self.event(done, Event::Begin(d, 4));
                        }
                    }
                }
            }
            Event::ZCopyDone(src, dst, e, start, width) => {
                trace!(
                    self,
                    json!({"event":"z_copy_done","cycle":self.now,"src":src,"dst":dst,
                    "task":e,"n_start":start,"n_valid":width})
                );
            }
            Event::Feedback(c, bin, q8) => {
                self.feedback_q8[c][bin] = q8;
                self.feedback_updates += 1;
                trace!(
                    self,
                    json!({"event":"service_feedback","cycle":self.now,"core":c,"bin":bin,"q8":q8}),
                );
            }
            Event::DrainDone(c) => {
                let s = self.cores[c].session.take().unwrap();
                if self.cfg.runtime_fsm {
                    let a = self
                        .dispatch_audit
                        .iter_mut()
                        .find(|a| {
                            a["task"].as_u64() == Some(s.e as u64)
                                && a["core"].as_u64() == Some(c as u64)
                        })
                        .unwrap();
                    let prediction = a["predicted_finish_cycle"].as_u64().unwrap();
                    a["actual_finish_cycle"] = json!(self.now);
                    a["actual_minus_predicted_cycles"] = json!(self.now as i64 - prediction as i64);
                }
                if self.feedback_enabled() && !s.split {
                    let bin = Self::feedback_bin(self.w.experts[s.e].m);
                    let sample =
                        ((self.now - s.start) * 256 / s.predicted_duration.max(1)).clamp(64, 1024);
                    let q8 = (3 * self.feedback_q8[c][bin] + sample + 2) / 4;
                    let cost = if self.cfg.control_cost { 2 } else { 0 };
                    let ready = self.now.max(self.control_free) + cost;
                    self.control_free = ready;
                    self.cores[c].stats.control_cycles += cost;
                    self.event(ready, Event::Feedback(c, bin, q8));
                }
                self.cores[c].stats.done_cycle = self.now;
                self.cores[c].stats.prediction_absolute_error_cycles +=
                    s.predicted_duration.abs_diff(self.now - s.start);
                self.cores[c].stats.prediction_actual_cycles += self.now - s.start;
                self.cores[c].stats.expert_completions += 1;
                self.cores[c].run = None;
                assert!(self.cores[c].incoming.is_none());
                if s.split {
                    self.split_drained[s.e][c] = true;
                    if self.split_drained[s.e].iter().all(|v| *v) {
                        self.status[s.e] = 2;
                    }
                } else {
                    self.status[s.e] = 2;
                }
                self.decision_needed = true;
                trace!(
                    self,
                    json!({"event":"expert_drained","cycle":self.now,"core":c,"expert":self.w.experts[s.e].id,"split":s.split})
                );
            }
        }
    }
    fn admit_weights(&mut self) {
        if self.now < self.control_free {
            return;
        }
        for off in 0..self.cores.len() {
            let c = (self.rr_desc + off) % self.cores.len();
            let eligible = self.cores[c]
                .run
                .as_ref()
                .is_some_and(|r| r.admit < r.tiles.len())
                && !self.cores[c].free_slots.is_empty();
            if !eligible {
                continue;
            }
            let slot = self.cores[c].free_slots.pop().unwrap();
            let cost = if self.cfg.control_cost { 2 } else { 0 };
            self.control_free = self.now + cost;
            self.cores[c].stats.control_cycles += cost;
            let tid = self.cores[c].run.as_ref().unwrap().admit;
            self.cores[c].live_tiles[slot] = Some(tid);
            let run = self.cores[c].run.as_mut().unwrap();
            let tile = &mut run.tiles[run.admit];
            tile.slot = Some(slot);
            tile.release = self.control_free;
            let bytes = tile.bytes as u64;
            run.admit += 1;
            self.cores[c].stats.weight_bytes += bytes;
            let used = self.cores[c].wslots - self.cores[c].free_slots.len();
            self.cores[c].stats.weight_peak_bytes =
                self.cores[c].stats.weight_peak_bytes.max(used * 4096);
            self.rr_desc = (c + 1) % self.cores.len();
            break;
        }
    }
    fn hbm_issue(&mut self) {
        if self.cfg.runtime_fsm {
            self.issue_runtime();
            return;
        }
        let grants = if self.cfg.ideal_hbm {
            self.cfg.credits
        } else {
            self.cfg.hbm_bytes_per_ns / 32
        };
        for _ in 0..grants {
            if self.credit_used >= self.cfg.credits {
                break;
            }
            let mut candidates = vec![];
            for c in 0..self.cores.len() {
                if let Some(run) = &self.cores[c].run {
                    let t = run.send_cursor;
                    if t < run.admit && run.tiles[t].release <= self.now {
                        assert!(run.tiles[t].sent < run.tiles[t].bytes);
                        candidates.push((c, t));
                    }
                }
            }
            if candidates.is_empty() {
                break;
            }
            let pick = if self.cfg.arbiter == "urgency" {
                *candidates
                    .iter()
                    .min_by_key(|&&(c, _)| {
                        let r = self.cores[c].run.as_ref().unwrap();
                        let ready = self.cores[c]
                            .live_tiles
                            .iter()
                            .flatten()
                            .filter(|&&t| r.tiles[t].ready)
                            .count();
                        // Tie rotates; periodic RR grant bounds starvation.
                        (
                            if self.now % 16 == 0 { 0 } else { ready },
                            (c + self.cores.len() - self.rr_hbm) % self.cores.len(),
                        )
                    })
                    .unwrap()
            } else {
                *candidates
                    .iter()
                    .min_by_key(|&&(c, _)| (c + self.cores.len() - self.rr_hbm) % self.cores.len())
                    .unwrap()
            };
            let (c, t) = pick;
            let tile = &mut self.cores[c].run.as_mut().unwrap().tiles[t];
            let offset = tile.sent;
            tile.sent += 32;
            if tile.sent == tile.bytes {
                self.cores[c].run.as_mut().unwrap().send_cursor += 1;
            }
            self.credit_used += 1;
            self.credit_peak = self.credit_peak.max(self.credit_used);
            self.rr_hbm = (c + 1) % self.cores.len();
            let latency = if self.cfg.ideal_hbm {
                0
            } else {
                self.cfg.hbm_latency_ns
            };
            self.event(self.now + latency, Event::Return(c, t, offset));
        }
    }
    fn prepare_x(&mut self, c: usize) {
        let Some(run) = &self.cores[c].run else {
            return;
        };
        if run.pos >= run.issues.len() {
            return;
        }
        let current = run.issues[run.pos].spec.clone();
        let tag = XTag {
            k: current.k_start,
            m: current.m_start,
        };
        let next = run.issues[run.pos..]
            .iter()
            .take(self.cfg.group + 1)
            .find(|i| i.spec.k_start != tag.k || i.spec.m_start != tag.m)
            .map(|i| i.spec.clone());
        let mut requested = vec![current];
        if let Some(i) = next {
            requested.push(i);
        }
        let keep: Vec<_> = requested
            .iter()
            .map(|s| XTag {
                k: s.k_start,
                m: s.m_start,
            })
            .collect();
        for spec in requested {
            let wanted = XTag {
                k: spec.k_start,
                m: spec.m_start,
            };
            if self.cores[c].xs.iter().any(|s| s.tag == Some(wanted)) {
                continue;
            }
            let slot = self.cores[c].xs.iter().position(|s| {
                s.busy_until <= self.now
                    && s.ready <= self.now
                    && s.tag.is_none_or(|t| !keep.contains(&t))
            });
            let Some(slot) = slot else {
                continue;
            };
            let bytes = spec.m_valid * spec.k_valid * 2;
            let s = self.cores[c].session.as_ref().unwrap();
            let name = if s.phase < 2 { "x" } else { "z" };
            let (base, stride) = self.allocation(c, name);
            let src = Self::spans(
                base + spec.m_start * stride,
                stride,
                spec.m_valid,
                spec.k_start * 2,
                spec.k_valid * 2,
            );
            let read = if self.cfg.ideal_onchip {
                self.now
            } else {
                self.cores[c].ab.access(self.now, &src, true, false)
            };
            let end = if self.cfg.ideal_onchip {
                self.now
            } else {
                let start = read.max(self.bus_free);
                self.bus_free = start + ceil(bytes, self.cfg.onchip_bytes_per_ns) as u64;
                let spans: Vec<_> = (0..spec.m_valid)
                    .map(|r| ((slot * self.cores[c].m + r) * 1024, spec.k_valid * 2))
                    .collect();
                self.cores[c].xb.access(self.bus_free, &spans, false, false)
            };
            self.cores[c].xs[slot] = XSlot {
                tag: Some(wanted),
                ready: end,
                busy_until: end,
            };
            self.cores[c].stats.x_stage_bytes += bytes as u64;
            if self.cfg.record_trace {
                let s = self.cores[c].session.as_ref().unwrap();
                trace!(
                    self,
                    json!({"event":"x_stage","cycle":self.now,"ready_cycle":end,
                    "core":c,"task":s.e,"phase":s.phase,"slot":slot,
                    "m_start":spec.m_start,"m_valid":spec.m_valid,
                    "k_start":spec.k_start,"k_valid":spec.k_valid})
                );
            }
            let count = self.cores[c].xs.iter().filter(|x| x.tag.is_some()).count();
            self.cores[c].stats.x_peak_bytes = self.cores[c]
                .stats
                .x_peak_bytes
                .max(count * self.cores[c].m * 1024);
        }
    }
    fn step_core(&mut self, c: usize) -> &'static str {
        if self.cores[c].session.is_none() {
            return "idle_no_expert";
        }
        if self.cores[c].run.is_none() {
            return "vector_copy_or_phase_barrier";
        }
        let (pos, len, chunk, next_chunk, converted, converting) = {
            let r = self.cores[c].run.as_ref().unwrap();
            (
                r.pos,
                r.issues.len(),
                r.chunk,
                r.issues.get(r.pos).map(|i| i.chunk),
                r.converted,
                r.converting,
            )
        };
        if next_chunk != Some(chunk) {
            if self.cores[c].contexts > 0 || self.now < self.cores[c].feed_until {
                return "chunk_result_drain";
            }
            if converting {
                return "chunk_conversion";
            }
            if !converted {
                let s = self.cores[c].session.as_ref().unwrap().clone();
                let n = if s.phase < 2 { s.fn_ } else { s.hn };
                let width = (n - chunk * self.cfg.group * 4).min(self.cfg.group * 4);
                let m = self.w.experts[s.e].m;
                let elems = m * width;
                let (scratch, _) = self.allocation(c, "scratch0");
                let mut ss = vec![];
                for band in 0..ceil(width, 4) {
                    for row in 0..m {
                        ss.push((
                            scratch + band * m * 32 + row * 32,
                            (width - band * 4).min(4) * 4,
                        ));
                    }
                }
                let end = if s.phase < 2 {
                    let name = if s.phase == 0 { "gate_z" } else { "up" };
                    let (b, st) = self.allocation(c, name);
                    let col = chunk * self.cfg.group * 4 + if s.phase == 0 { s.f0 } else { 0 };
                    let ds = Self::spans(b, st, m, col * 2, width * 2);
                    self.vector_io(c, elems, &ss, &ds)
                } else {
                    // Copy finished scratch outputs into producer-Y workspace, release metadata.
                    let (b, st) = self.allocation(c, "y");
                    let ds = Self::spans(b, st, m, chunk * self.cfg.group * 4 * 4, width * 4);
                    self.transfer_io(c, c, self.now, &ss, &ds, false)
                };
                self.cores[c].run.as_mut().unwrap().converting = true;
                self.event(end, Event::ChunkConverted(c));
                return "chunk_conversion";
            }
            if pos == len {
                self.end_projection(c);
                return "phase_advance";
            }
            let r = self.cores[c].run.as_mut().unwrap();
            r.chunk = next_chunk.unwrap();
            r.converted = false;
            // Prior N group's K updates have drained; its bounded scoreboard
            // shares the released group metadata, never grows with full N.
            r.committed.clear();
        }
        self.prepare_x(c);
        if self.now < self.cores[c].feed_until {
            return "operand_feed";
        }
        let (i, spec, tid, ready, slot, previous) = {
            let r = self.cores[c].run.as_ref().unwrap();
            let i = r.pos;
            let m = &r.issues[i];
            let t = &r.tiles[m.tile];
            (
                i,
                m.spec.clone(),
                m.tile,
                t.ready,
                t.slot,
                *r.committed
                    .get(&(m.spec.m_start, m.spec.n_start))
                    .unwrap_or(&0),
            )
        };
        if !ready {
            if self.cfg.runtime_fsm && tid == 0 && self.cores[c].prefetch_promoted_at.is_some() {
                self.cores[c].stats.next_weight_wait_cycles += 1;
            }
            return "weight_not_ready";
        }
        if previous != spec.k_start / 512 {
            return "previous_k_commit";
        }
        if self.cores[c].contexts >= 8 {
            return "result_context_full";
        }
        let tag = XTag {
            k: spec.k_start,
            m: spec.m_start,
        };
        let Some(xs) = self.cores[c]
            .xs
            .iter()
            .position(|x| x.tag == Some(tag) && x.ready <= self.now && x.busy_until <= self.now)
        else {
            return "x_not_ready";
        };
        let wspans: Vec<_> = (0..spec.n_valid)
            .map(|n| (slot.unwrap() * 4096 + n * 1024, spec.k_valid * 2))
            .collect();
        let xspans: Vec<_> = (0..spec.m_valid)
            .map(|m| ((xs * self.cores[c].m + m) * 1024, spec.k_valid * 2))
            .collect();
        let end = if self.cfg.ideal_onchip {
            self.now + 1
        } else {
            // Reuse is in the charged W SRAM. No extra persistent 4 KiB operand cache.
            let wr = self.cores[c].wb.access(self.now, &wspans, true, false);
            wr.max(self.cores[c].xb.access(self.now, &xspans, true, false))
                .max(self.now + 1)
        };
        let e = self.cores[c].session.as_ref().unwrap().e;
        let last = spec.m_start + spec.m_valid == self.w.experts[e].m;
        self.cores[c].xs[xs].busy_until = end;
        self.cores[c].feed_until = end;
        self.cores[c].weight_cache = Some(tid);
        self.cores[c].arithmetic_until = end + self.cfg.dot_tail_ns;
        self.cores[c].contexts += 1;
        self.cores[c].stats.contexts_peak = self.cores[c]
            .stats
            .contexts_peak
            .max(self.cores[c].contexts);
        self.cores[c].stats.issues += 1;
        self.profile_issue(c, i, tid, xs, end);
        self.last_progress = self.now;
        if self.cfg.record_trace {
            let session = self.cores[c].session.as_ref().unwrap();
            trace!(
                self,
                json!({"event":"mac_issue","cycle":self.now,"core":c,"task":session.e,
                "phase":session.phase,"issue":i,"tile":tid,"slot":slot,"x_slot":xs,
                "m_start":spec.m_start,"m_valid":spec.m_valid,"n_start":spec.n_start,
                "n_valid":spec.n_valid,"k_start":spec.k_start,"k_valid":spec.k_valid}),
            );
        }
        self.cores[c].stats.useful_macs += (spec.m_valid * spec.n_valid * spec.k_valid) as u64;
        self.cores[c].stats.issued_macs += (self.cores[c].m * 4 * 512) as u64;
        self.cores[c].run.as_mut().unwrap().pos += 1;
        self.event(end, Event::Feed(c, tid, xs, last));
        self.event(end + self.cfg.dot_tail_ns, Event::Dot(c, i));
        "issue"
    }
    fn new(w: Workload, cfg: Config) -> Self {
        assert!(
            matches!(
                cfg.lanes.as_slice(),
                [6] | [3, 3] | [4, 2] | [5, 1] | [8] | [4, 4] | [5, 3] | [6, 2] | [7, 1]
            ),
            "unsupported spatial organization"
        );
        assert!(cfg.onchip_bytes_per_ns > 0 && cfg.vector_elements_per_ns > 0);
        assert!(cfg.max_cycles > 0 && cfg.no_progress_cycles > 0);
        assert!(matches!(
            cfg.dispatch.as_str(),
            "fixed" | "fifo" | "dynamic" | "feedback" | "joint"
        ));
        assert!(matches!(cfg.arbiter.as_str(), "rr" | "urgency" | "stock"));
        assert!(cfg.joint_age_limit > 0);
        if cfg.dispatch == "joint" {
            assert!(cfg.runtime_fsm && cfg.split == "none");
            assert!(cfg.late_bind_cycles.is_none() && cfg.surplus_rules == 0 && !cfg.shared_large);
        }
        assert!(cfg.next_prefetch_ready_threshold.is_none_or(|n| n > 0));
        assert!(cfg.late_bind_cycles.is_none_or(|n| n > 0));
        assert!(cfg.surplus_rules <= 4);
        if cfg.surplus_rules > 0 {
            assert!(cfg.runtime_fsm && cfg.split == "none");
            assert!(
                cfg.late_bind_cycles.is_none()
                    && cfg.next_prefetch_ready_threshold.is_none()
                    && !cfg.shared_large
            );
            assert_eq!(
                w.engine_layout["surplus_lut"]["cores"]
                    .as_array()
                    .unwrap()
                    .len(),
                cfg.lanes.len()
            );
        }
        assert!(
            cfg.runtime_fsm
                || (cfg.next_prefetch_ready_threshold.is_none()
                    && cfg.late_bind_cycles.is_none()
                    && !cfg.shared_large
                    && cfg.arbiter != "stock")
        );
        assert!(w.hidden > 0 && w.batch > 0);
        for (index, e) in w.experts.iter().enumerate() {
            assert!(
                e.m > 0 && e.h == w.hidden && e.f > 0,
                "invalid expert geometry"
            );
            assert!(
                e.m.checked_mul(e.h)
                    .and_then(|n| n.checked_mul(e.f))
                    .and_then(|n| n.checked_mul(6))
                    .is_some(),
                "expert geometry overflows"
            );
            assert!(e.token_indices.len() == e.m && e.token_indices.iter().all(|&t| t < w.batch));
            assert!(
                !w.experts[..index].iter().any(|other| other.id == e.id),
                "duplicate expert id"
            );
        }
        assert!(
            matches!(cfg.group, 1 | 2 | 4)
                && cfg.window > 0
                && cfg.window <= 8
                && cfg.hbm_bytes_per_ns >= 32
                && cfg.credits > 0
                && cfg.credits
                    <= if cfg.diagnostic_credit_expansion {
                        4096
                    } else {
                        256
                    }
                && cfg.dma_ready_period > 0
                && cfg.dma_ready_cycles > 0
                && cfg.dma_ready_cycles <= cfg.dma_ready_period
        );
        let nc = cfg.lanes.len();
        let total_m: usize = cfg.lanes.iter().sum();
        let hw = &w.engine_layout["hardware"];
        if !matches!(cfg.lanes.as_slice(), [6] | [3, 3] | [4, 2]) {
            assert!(
                !hw.is_null(),
                "new organizations require explicit compiler budgets"
            );
        }
        if !hw.is_null() {
            for (field, total) in [
                ("weight_slots", 10),
                ("weight_banks", 64),
                ("x_banks", 4 * total_m),
                ("acc_banks", 2 * total_m),
                ("acc_bytes", 2 * 1024 * 1024),
                (
                    "control_bytes",
                    4096 + hw["joint_state_bytes"].as_u64().unwrap_or(0) as usize,
                ),
            ] {
                let v = hw[field].as_array().expect("missing physical budget");
                assert_eq!(v.len(), nc);
                assert!(v.iter().all(|x| x.as_u64().is_some_and(|v| v > 0)));
                assert_eq!(
                    v.iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .sum::<usize>(),
                    total
                );
            }
        }
        if cfg.dispatch == "joint" {
            assert_eq!(hw["joint_state_bytes"], 256);
        }
        if cfg.dispatch == "feedback"
            || cfg.tail_partition
            || (cfg.dispatch == "joint" && cfg.joint_feedback)
        {
            assert!(cfg.runtime_fsm && cfg.split == "none");
            assert_eq!(hw["feedback_state_bytes"], 96);
        }
        let mut cores = vec![];
        let rows: usize = w.experts.iter().map(|e| e.m).sum();
        let mut before = 0;
        for &m in &cfg.lanes {
            let lo = ceil(w.hidden, 4) * before / total_m * 4;
            let hi = (ceil(w.hidden, 4) * (before + m) / total_m * 4).min(w.hidden);
            let result = (hi - lo) * (4 * rows + 4 * w.batch + 2 * w.batch);
            let route_state = w.batch * w.top_k * 16 + w.experts.len() * 64;
            let idx = cores.len();
            let reserved = if w.engine_layout.is_null() {
                align(result + ceil(4096 * m, total_m) + route_state, 32)
            } else {
                w.engine_layout["cores"][idx]["reserved"].as_u64().unwrap() as usize
            };
            let capacity = if w.engine_layout.is_null() {
                2 * 1024 * 1024 * m / total_m
            } else {
                w.engine_layout["cores"][idx]["capacity"].as_u64().unwrap() as usize
            };
            let resource = |field: &str, fallback: usize| -> usize {
                hw[field][idx].as_u64().map_or(fallback, |x| x as usize)
            };
            let slots = resource("weight_slots", 10 / nc);
            assert!(slots > cfg.group, "no slot left for bounded lookahead");
            cores.push(Core {
                m,
                wslots: slots,
                wb: Banks::new(resource("weight_banks", 64 / nc)),
                xb: Banks::new(resource("x_banks", 4 * m)),
                ab: Banks::new(resource("acc_banks", 2 * m)),
                session: None,
                next: None,
                incoming: None,
                ahead: None,
                surplus_thresholds: None,
                prefetch_promoted_at: None,
                dma_wait_since: None,
                run: None,
                xs: (0..2)
                    .map(|_| XSlot {
                        tag: None,
                        ready: 0,
                        busy_until: 0,
                    })
                    .collect(),
                free_slots: (0..slots).collect(),
                live_tiles: vec![None; slots],
                contexts: 0,
                feed_until: 0,
                arithmetic_until: 0,
                blocked_until: 0,
                weight_cache: None,
                reserved,
                capacity,
                stats: Stats::default(),
            });
            before += m;
        }
        let ne = w.experts.len();
        let mut sim = Self {
            time_origin: 0,
            w,
            cfg,
            cores,
            now: 0,
            seq: 0,
            events: BinaryHeap::new(),
            credit_used: 0,
            credit_peak: 0,
            control_free: 0,
            bus_free: 0,
            vector_free: 0,
            rr_desc: 0,
            rr_hbm: 0,
            status: vec![0; ne],
            fixed: vec![0; ne],
            split_ready: vec![vec![false; nc]; ne],
            split_drained: vec![vec![false; nc]; ne],
            trace: vec![],
            decisions: 0,
            deferrals: 0,
            decision_needed: true,
            decision_after: 0,
            decision_cursor: 0,
            all_done_at: None,
            combine_cycles: 0,
            unassigned_peak: 0,
            pending: VecDeque::new(),
            input_cursor: 0,
            pending_dma: None,
            outstanding_dma: BTreeMap::new(),
            dma_serial: 0,
            last_progress: 0,
            runtime_decision_paid: false,
            runtime_candidate_mask: 0,
            late_bind_wait_cycles: 0,
            prefetch_gate_wait: [0; 2],
            promotion_pending: vec![None; nc],
            input_backpressure_cycles: 0,
            dispatch_audit: vec![],
            diagnostics: surplus::Diagnostics::new(nc),
            feedback_q8: vec![[256; 4]; nc],
            feedback_updates: 0,
            tail_decision_ready: None,
            tail_partition_count: 0,
            joint: joint::JointState::default(),
            profile_v3: profile_v3::Profile::new(nc),
        };
        let mut load = vec![0u64; nc];
        for e in 0..ne {
            let c = (0..nc)
                .filter(|&c| sim.fits(e, c, false))
                .min_by_key(|&c| load[c] + sim.prediction(e, c, nc, false))
                .expect("no legal whole-expert core");
            sim.fixed[e] = c;
            load[c] += sim.prediction(e, c, nc, false);
        }
        if !sim.cfg.fixed_assignment.is_empty() {
            assert_eq!(sim.cfg.dispatch, "fixed");
            assert_eq!(sim.cfg.fixed_assignment.len(), ne);
            for (e, &c) in sim.cfg.fixed_assignment.iter().enumerate() {
                assert!(c < nc && sim.fits(e, c, false), "illegal oracle owner");
            }
            sim.fixed.clone_from(&sim.cfg.fixed_assignment);
        }
        if sim.cfg.dispatch == "fixed" && sim.cfg.control_cost {
            let cost = (2 * ne * nc) as u64;
            sim.control_free = cost;
            sim.cores[0].stats.control_cycles += cost;
        }
        sim
    }
    fn run(mut self) -> Value {
        loop {
            // Same-cycle completions are drained before a new dispatch/issue decision.
            while self
                .events
                .peek()
                .is_some_and(|Reverse((t, _, _))| *t <= self.now)
            {
                let Reverse((t, _, e)) = self.events.pop().unwrap();
                assert_eq!(t, self.now);
                self.handle(e);
            }
            if self.status.iter().all(|&s| s == 2) && self.events.is_empty() {
                assert_eq!(self.credit_used, 0);
                self.all_done_at = Some(self.now);
                break;
            }
            if self.cfg.runtime_fsm {
                // Commit a completed control operation before another actor
                // acquires the port. Its completion is not the port's next free time.
                self.finish_runtime_promotions();
            }
            self.dispatch();
            if self.cfg.runtime_fsm {
                self.promote_runtime();
            }
            self.admit_weights();
            if self.cfg.runtime_fsm {
                self.admit_phase_ahead();
                self.admit_next_weights();
            }
            self.diagnostics.accepted_this_cycle.fill(false);
            self.hbm_issue();
            self.profile_hbm_cycle();
            for c in 0..self.cores.len() {
                let state = self.step_core(c);
                self.profile_core_cycle(c, state);
                self.diagnostics.front_wait_code[c] = match state {
                    "previous_k_commit" => 1,
                    "x_not_ready" => 2,
                    "weight_not_ready" => 3,
                    "result_context_full" => 4,
                    "chunk_result_drain" => 5,
                    "chunk_conversion" => 6,
                    "operand_feed" => 7,
                    _ => 0,
                };
                *self.cores[c]
                    .stats
                    .front_states
                    .entry(state.into())
                    .or_default() += 1;
                if self.now < self.cores[c].arithmetic_until {
                    self.cores[c].stats.arithmetic_active_cycles += 1;
                }
            }
            if self.cfg.runtime_fsm {
                self.observe_supply();
            }
            // Zero-latency oracle events can be enqueued by this cycle's service.
            while self
                .events
                .peek()
                .is_some_and(|Reverse((t, _, _))| *t == self.now)
            {
                let Reverse((_, _, e)) = self.events.pop().unwrap();
                self.handle(e);
            }
            self.now += 1;
            assert!(
                self.now < self.cfg.max_cycles,
                "model progress timeout: {}",
                self.wait_snapshot()
            );
            assert!(
                self.now.saturating_sub(self.last_progress) < self.cfg.no_progress_cycles,
                "no progress: {}",
                self.wait_snapshot()
            );
        }
        let experts_done = self.now;
        let rows: usize = self.w.experts.iter().map(|e| e.m).sum();
        // Fixed router order within each output-column partition. Shared vector service.
        let mut done = self.now;
        for c in 0..self.cores.len() {
            let (_, n) = self.bounds(self.w.hidden, c);
            let mut ss = vec![];
            for e in 0..self.w.experts.len() {
                let (b, st) = self.result_address(c, "inbox", e);
                ss.extend(Self::spans(b, st, self.w.experts[e].m, 0, n * 4));
            }
            let (b, st) = self.result_address(c, "combined_output", 0);
            let ds = Self::spans(b, st, self.w.batch, 0, n * 4);
            done = done.max(self.vector_io(c, rows * n + 2 * self.w.batch * n, &ss, &ds));
        }
        self.combine_cycles = done - experts_done;
        self.now = done;
        self.profile_combine_tail(experts_done, done);
        let expected: u64 = self
            .w
            .experts
            .iter()
            .map(|e| (3 * e.m * e.h * e.f) as u64)
            .sum();
        let useful: u64 = self.cores.iter().map(|c| c.stats.useful_macs).sum();
        assert_eq!(expected, useful);
        let expected_w: u64 = self
            .w
            .experts
            .iter()
            .map(|e| (2 * e.f * align(e.h * 2, 32) + e.h * align(e.f * 2, 32)) as u64)
            .sum();
        let weight: u64 = self.cores.iter().map(|c| c.stats.weight_bytes).sum();
        assert_eq!(expected_w, weight);
        for c in &self.cores {
            assert_eq!(c.free_slots.len(), c.wslots);
            assert_eq!(c.contexts, 0);
            assert!(c.stats.workspace_peak_bytes <= c.capacity);
            assert!(c.next.is_none() && c.incoming.is_none() && c.session.is_none());
            assert!(c.ahead.is_none());
        }
        assert!(self.outstanding_dma.is_empty() && self.pending_dma.is_none());
        if self.cfg.runtime_fsm {
            assert!(self.pending.is_empty() && self.input_cursor == self.w.experts.len());
            assert_eq!(self.dma_serial * 32, weight);
        }
        let mut report = json!({"schema":"plena_dispatch_analytical_v1","workload":self.w.id,"config":self.cfg,
   "scope":"routed MoE from resident inputs/routes to ordered combine; analytical HBM, not Ramulator; timing metadata, separate payload tests",
   "cycles":self.now,"time_ms_at_1ghz":self.now as f64/1e6,"experts_done_cycles":experts_done,"combine_tail_cycles":self.combine_cycles,
   "useful_macs":useful,"issued_macs":self.cores.iter().map(|c|c.stats.issued_macs).sum::<u64>(),
   "weight_bytes":weight,"credit_peak":self.credit_peak,"dispatch_decisions":self.decisions,"deferrals":self.deferrals,
   "pending_window_peak":self.unassigned_peak,"drained":true,"ownership_k_order_capacity_checks":true,
   "dma_transactions_accepted":self.dma_serial,"dma_transactions_landed":self.cores.iter().map(|c|c.stats.dma_landed).sum::<u64>(),
   "dma_credit_bytes":32,"dma_return_capacity_bytes":8192.max(self.cfg.credits*32),"input_backpressure_cycles":self.input_backpressure_cycles,
   "resource_conditions":{"over_budget_credit_diagnostic":self.cfg.diagnostic_credit_expansion,
     "extra_return_bytes":self.cfg.credits.saturating_sub(256)*32,
     "extra_credit_tag_bytes":self.cfg.credits.saturating_sub(256)*2,
     "not_eligible_for_equal_budget_claim":self.cfg.credits>256},
   "dispatch_audit":self.dispatch_audit,
   "supply_diagnostics":self.diagnostics,
   "feedback_q8":self.feedback_q8,"feedback_updates":self.feedback_updates,
   "tail_partition_count":self.tail_partition_count,
   "joint_diagnostics":self.joint.diagnostics,
   "physical_budget":self.w.engine_layout.get("hardware"),
   "late_bind_wait_cycles":self.late_bind_wait_cycles,
   "prefetch_gate_wait_core_cycles": {"current_requests_unsent":self.prefetch_gate_wait[0], "ready_tiles_at_or_above_threshold":self.prefetch_gate_wait[1]},
   "control_accounting":self.w.engine_layout.get("control_accounting"),
   "numerical_validation":"separate address-payload tests; this timing run does not execute numerical tensors",
   "model_limitations":["HBM is aggregate bandwidth plus fixed response latency, not channel/row timing", "vector and copy phases are conservatively charged as read/service/write without overlap", "candidate scalar cost heuristic, not optimal dispatch", "timing/payload are separately verified, not a unified native emulator"],
   "cores":self.cores.iter().map(|c|json!({"m":c.m,"capacity":c.capacity,"reserved_input_result_control":c.reserved,"stats":c.stats,
    "weight_bank_words":c.wb.words,"x_bank_words":c.xb.words,"workspace_bank_words":c.ab.words})).collect::<Vec<_>>(),"trace":self.trace});
        if self.cfg.diagnostic_profile {
            report["m0_profile"] = self.profile_v3.report(self.now);
        }
        report
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: binary workload.json config.json output.json"
    );
    if args[1] == "--compute" {
        let input: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
        let report = compute::run(&input);
        std::fs::write(&args[3], serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        return;
    }
    if args[1] == "--v3-replay" {
        let input: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
        let report = v3::replay(&input).expect("v3 numerical replay failed");
        std::fs::write(&args[3], serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        return;
    }
    let workload_value: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let mut config_value: Value =
        serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    if config_value["arch"] == "supply_v3" {
        let numeric_payload = config_value.get("numeric_payload")
            .or_else(|| workload_value.get("numeric_payload"))
            .cloned().unwrap_or(Value::Null);
        let report = v3::run(&json!({"workload":workload_value,"config":config_value,"numeric_payload":numeric_payload}))
            .expect("supply_v3 failed");
        std::fs::write(&args[3], serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        return;
    }
    let original_config = config_value.clone();
    let scale = config_value["weight_bytes_scale"].as_f64().unwrap_or(1.0);
    assert!(
        scale > 0.0 && scale <= 1.0,
        "weight byte timing oracle must be in (0,1]"
    );
    // Diagnostic-only uniform compression oracle. Logical BF16 transfers and
    // all landing/read/compute costs remain unchanged. Scale shared wire service
    // and effective wire-credit occupancy together, never SRAM allocations.
    if scale < 1.0 {
        let bw = config_value["hbm_bytes_per_ns"].as_u64().unwrap_or(256);
        let cr = config_value["credits"].as_u64().unwrap_or(256);
        config_value["hbm_bytes_per_ns"] = json!((bw as f64 / scale).floor() as u64);
        config_value["credits"] = json!((cr as f64 / scale).ceil() as u64);
        config_value["diagnostic_credit_expansion"] = json!(true);
    }
    let cfg: Config = serde_json::from_value(config_value).unwrap();
    let mut report = if workload_value.get("legacy_batch_execution").is_some() {
        legacy_chunks::run(&workload_value, cfg)
    } else {
        let w: Workload = serde_json::from_value(workload_value).unwrap();
        Sim::new(w, cfg).run()
    };
    if scale < 1.0 {
        report["compression_oracle"] = json!({"weight_bytes_scale":scale,
            "original_config":original_config,"logical_weight_bytes":report["weight_bytes"],
            "scaled_wire_bytes":(report["weight_bytes"].as_u64().unwrap() as f64 * scale).ceil() as u64,
            "scope":"nonphysical uniform wire-service/credit scaling; same logical BF16 SRAM moves and issued operations; no dequant cost; not packed P1/P2"});
    }
    std::fs::write(&args[3], serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn workload(ms: &[usize]) -> Workload {
        Workload {
            id: "test".into(),
            batch: *ms.iter().max().unwrap(),
            hidden: 512,
            top_k: 1,
            engine_layout: Value::Null,
            experts: ms
                .iter()
                .enumerate()
                .map(|(id, &m)| Expert {
                    id: id as i64,
                    is_shared: false,
                    m,
                    h: 512,
                    f: 16,
                    token_indices: (0..m).collect(),
                    weights: Value::Null,
                })
                .collect(),
        }
    }
    #[test]
    fn whole_and_split_preserve_work_and_bytes() {
        let w = workload(&[4, 2]);
        let a = Sim::new(w.clone(), Config::default()).run();
        let mut cfg = Config::default();
        cfg.split = "forced".into();
        cfg.runtime_fsm = false;
        cfg.arbiter = "rr".into();
        let b = Sim::new(w, cfg).run();
        assert_eq!(a["weight_bytes"], b["weight_bytes"]);
        assert_eq!(a["useful_macs"], b["useful_macs"]);
    }
    #[test]
    fn repeat_is_exact() {
        let a = Sim::new(workload(&[3, 3]), Config::default()).run();
        let b = Sim::new(workload(&[3, 3]), Config::default()).run();
        assert_eq!(a, b);
    }
    #[test]
    fn all_organizations_drain() {
        for lanes in [vec![6], vec![3, 3], vec![4, 2]] {
            let cfg = Config {
                lanes,
                ..Default::default()
            };
            let r = Sim::new(workload(&[1, 4, 2]), cfg).run();
            assert_eq!(r["drained"], true);
        }
    }
}
