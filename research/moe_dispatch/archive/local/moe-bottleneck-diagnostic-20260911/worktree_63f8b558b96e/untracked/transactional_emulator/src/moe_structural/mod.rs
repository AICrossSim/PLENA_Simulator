//! Max-plus resource scheduling with explicit banks and folded dot-product beats.
//! Each service is reserved causally, with finite W/X slots and ordered output RMW.
use crate::moe_spatial::fabric::{WeightRead, WeightSource};
use crate::moe_spatial::{self, Request, operands::Operands};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

mod x_reuse;
use x_reuse::{ReuseCursor, TilePlan, group_tiles};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Hardware {
    pub parallel_k: usize,
    pub mul_latency: u64,
    pub add_latency: u64,
    pub core_period_ps: u64,
    pub weight_banks_total: usize,
    pub activation_banks_per_m: usize,
    pub accumulator_banks_per_m: usize,
    pub bank_word_bytes: usize,
    pub bank_read_latency: u64,
    pub weight_bytes_total: usize,
    pub activation_bytes_total: usize,
    pub accumulator_bytes_total: usize,
    pub x_slots_per_core: usize,
    pub result_contexts_per_core: usize,
    pub band_window: usize,
    pub control_word_bytes: usize,
    pub descriptor_bytes: usize,
    pub source: String,
    pub assignment: String,
    pub native: moe_spatial::native::Config,
    pub record_trace: bool,
    /// Zero preserves the frozen tile-major policy. Otherwise interleave the
    /// M blocks of at most this many same-expert, same-K resident weight tiles.
    #[serde(default)]
    pub x_reuse_bands: usize,
}

#[derive(Clone, Serialize, Debug)]
struct Banks {
    free: Vec<u64>,
    word: usize,
    read_latency: u64,
    word_services: u64,
    port_wait_cycles: u64,
}
struct StreamBus {
    cycle: u64,
    used: usize,
    lanes: usize,
}
impl Banks {
    fn new(count: usize, word: usize, latency: u64) -> Self {
        Self {
            free: vec![0; count],
            word,
            read_latency: latency,
            word_services: 0,
            port_wait_cycles: 0,
        }
    }
    // One shared read/write port per bank. A command owns each touched bank's
    // FIFO positions. Read latency does not occupy the command port after issue.
    fn access(&mut self, now: u64, spans: &[(usize, usize)], read: bool) -> u64 {
        let mut counts = vec![0_u64; self.free.len()];
        let mut last = None;
        for &(address, bytes) in spans {
            if bytes == 0 {
                continue;
            }
            for word in address / self.word..(address + bytes).div_ceil(self.word) {
                if last != Some(word) {
                    counts[word % self.free.len()] += 1;
                }
                last = Some(word);
            }
        }
        let mut end = now;
        for (bank, &count) in counts.iter().enumerate().filter(|(_, n)| **n > 0) {
            let start = now.max(self.free[bank]);
            self.port_wait_cycles += start - now;
            self.free[bank] = start + count;
            self.word_services += count;
            end = end.max(start + count + if read { self.read_latency - 1 } else { 0 });
        }
        end
    }
    // Explicit atomic RMW policy: bank remains locked across the dependent
    // FP32 add. Other banks may progress; no uncharged read/write ports.
    fn rmw(&mut self, now: u64, spans: &[(usize, usize)], add: u64) -> u64 {
        let mut counts = vec![0_u64; self.free.len()];
        for &(a, b) in spans {
            for w in a / self.word..(a + b).div_ceil(self.word) {
                counts[w % self.free.len()] += 1;
            }
        }
        let mut end = now;
        for (i, &n) in counts.iter().enumerate().filter(|(_, n)| **n > 0) {
            let start = now.max(self.free[i]);
            self.port_wait_cycles += start - now;
            self.free[i] = start + n * (self.read_latency + add + 1);
            self.word_services += 2 * n;
            end = end.max(self.free[i]);
        }
        end
    }
    /// On-chip producer -> one-cycle bus register -> private bank. Producer is
    /// backpressured before issuing a word; there is no invisible full-X buffer.
    fn stream_write(&mut self, now: u64, spans: &[(usize, usize)], bus: &mut StreamBus) -> u64 {
        let mut end = now;
        for &(a, b) in spans {
            if b == 0 {
                continue;
            }
            for w in a / self.word..(a + b).div_ceil(self.word) {
                let bank = w % self.free.len();
                let earliest = now.max(self.free[bank].saturating_sub(1));
                if bus.cycle < earliest {
                    bus.cycle = earliest;
                    bus.used = 0;
                }
                if bus.used == bus.lanes {
                    bus.cycle += 1;
                    bus.used = 0;
                }
                let arrive = bus.cycle + 1;
                assert!(arrive >= self.free[bank]);
                self.free[bank] = arrive + 1;
                self.word_services += 1;
                bus.used += 1;
                end = end.max(arrive + 1);
            }
        }
        end
    }
}

#[derive(Clone, Debug)]
struct Band {
    job: usize,
    nt: usize,
    core: usize,
    offset: usize,
}
#[derive(Clone, Debug)]
struct Tile {
    band: usize,
    kt: usize,
    slot: usize,
    ready: Option<u64>,
    row: usize,
    consumed_rows: usize,
    landing_done: u64,
    sequence: usize,
    group_len: usize,
}
#[derive(Clone, Debug)]
struct Task {
    tile: usize,
    row: usize,
    rows: usize,
    xslot: usize,
    x_ready: u64,
    group: usize,
    feed_ready: u64,
    started: bool,
}
#[derive(Clone, Debug)]
enum Event {
    SectorAck,
    Source(usize),
    Feed(usize, usize),
    Dot(usize, usize, usize, usize),
    Commit(usize, usize, usize, usize),
}

#[derive(Serialize, Default)]
struct CoreReport {
    feed_groups: u64,
    invocations: u64,
    useful_macs: u64,
    padded_macs: u64,
    weight_peak_bytes: usize,
    activation_peak_bytes: usize,
    result_context_peak: usize,
    accumulator_bytes: usize,
    done_cycle: u64,
    arithmetic_active_cycles: u64,
    first_feed_gap_min: Option<u64>,
    first_feed_gap_max: u64,
    first_feed_gap_sum: u64,
    issue_states: BTreeMap<String, u64>,
    x_cache_hits: u64,
    x_cache_misses: u64,
    x_input_bytes: u64,
    x_reuse_busy_tag_cycles: u64,
    weight_transport_wait_cycles: u64,
    weight_landing_wait_cycles: u64,
}
#[derive(Serialize)]
pub struct Report {
    schema: String,
    scope: String,
    source: String,
    hardware: Hardware,
    cycles: u64,
    time_ms_at_assumed_clock: f64,
    resource_bill: serde_json::Value,
    cores: Vec<CoreReport>,
    weight_banks: Vec<Banks>,
    activation_banks: Vec<Banks>,
    accumulator_banks: Vec<Banks>,
    source_bytes: u64,
    control_cycles: u64,
    activation_input_bytes: u64,
    useful_macs: u64,
    padded_macs: u64,
    numerical_bit_exact: Option<bool>,
    output_fp32_bits: Option<Vec<Vec<u32>>>,
    drained: bool,
    ownership_and_k_order_verified: bool,
    trace: Vec<serde_json::Value>,
    native: Option<serde_json::Value>,
}
fn log2(v: usize) -> u64 {
    usize::BITS as u64 - v.leading_zeros() as u64 - 1
}
fn ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn reuse_state_bytes(r: &Request, h: &Hardware) -> usize {
    if h.x_reuse_bands == 0 {
        0
    } else {
        // Four 64-bit sequencer registers/core and two 32-bit tags/W slot.
        // Observer counters are simulator instrumentation, not hardware state.
        32 * r.m_lanes.len()
            + 8 * h
                .weight_bytes_total
                .saturating_sub(h.native.outstanding_sectors * 32)
                / 4096
    }
}

fn validate(r: &Request, h: &Hardware) -> Result<(), String> {
    if r.m_lanes.is_empty()
        || r.m_lanes.iter().sum::<usize>() != 6
        || r.n_lanes != 4
        || r.k_lanes != 512
        || !h.parallel_k.is_power_of_two()
        || h.parallel_k > 512
        || h.mul_latency == 0
        || h.add_latency == 0
        || h.bank_read_latency == 0
        || h.weight_banks_total == 0
        || !h.weight_banks_total.is_multiple_of(r.m_lanes.len())
        || h.activation_banks_per_m == 0
        || h.accumulator_banks_per_m == 0
        || !h.bank_word_bytes.is_power_of_two()
        || h.bank_word_bytes < 4
        || h.x_slots_per_core == 0
        || h.result_contexts_per_core == 0
        || h.band_window == 0
        || h.control_word_bytes == 0
        || h.descriptor_bytes < 32
        || h.core_period_ps == 0
        || !h.weight_bytes_total.is_multiple_of(r.m_lanes.len() * 4096)
        || !matches!(
            h.source.as_str(),
            "native" | "onchip_oracle" | "compute_oracle"
        )
        || !matches!(h.assignment.as_str(), "bands" | "experts")
    {
        return Err("invalid structural contract; dimensions are M x N_tile=4 x K_tile=512".into());
    }
    if r.m_lanes.contains(&0)
        || r.jobs.is_empty()
        || r.jobs.iter().any(|j| j.m == 0 || j.n == 0 || j.k == 0)
    {
        return Err("empty core or GEMM dimension".into());
    }
    let control_state = r.m_lanes.len()
        * (h.band_window * 16 + h.result_contexts_per_core * 32 + h.x_slots_per_core * 32 + 64)
        + h.weight_bytes_total / 4096 * 32
        + 64
        + reuse_state_bytes(r, h);
    if control_state > 4096 {
        return Err("controller state exceeds reserved 4 KiB".into());
    }
    if h.x_slots_per_core * 6 * 512 * 2 > h.activation_bytes_total {
        return Err("X slots exceed SRAM".into());
    }
    if h.native.core_period_ps != h.core_period_ps {
        return Err("native and core clocks disagree".into());
    }
    Ok(())
}

// Offline compiler mapping, not an oracle runtime scheduler. No measured future
// HBM latency is used. Mapping is identical for all memory/primitive-latency runs.
fn map_bands(r: &Request, h: &Hardware) -> Result<Vec<Band>, String> {
    let nc = r.m_lanes.len();
    let mut load = vec![0_u64; nc];
    let mut bytes = vec![0_usize; nc];
    let budget: Vec<_> = r
        .m_lanes
        .iter()
        .map(|m| h.accumulator_bytes_total * m / 6)
        .collect();
    let mut bands = Vec::new();
    for (j, job) in r.jobs.iter().enumerate() {
        let mut pinned = None;
        for nt in 0..ceil(job.n, 4) {
            let needed = job.m * 32; // 16 B data +16 B persistent output metadata per row.
            let cost: Vec<_> = r
                .m_lanes
                .iter()
                .map(|m| (ceil(job.m, *m) * ceil(job.k, 512)) as u64)
                .collect();
            let candidate = (0..nc)
                .filter(|&c| {
                    pinned.is_none_or(|p| p == c)
                        && bytes[c] + needed + 4096 * r.m_lanes[c] / 6 <= budget[c]
                        && (h.assignment != "experts"
                            || pinned.is_some()
                            || bytes[c] + needed * ceil(job.n, 4) + 4096 * r.m_lanes[c] / 6
                                <= budget[c])
                })
                .min_by_key(|&c| (load[c] + cost[c], cost[c] * r.m_lanes[c] as u64, c));
            let c = candidate.ok_or("private output storage cannot admit fixed mapping")?;
            if h.assignment == "experts" {
                pinned = Some(c);
            }
            bands.push(Band {
                job: j,
                nt,
                core: c,
                offset: bytes[c],
            });
            bytes[c] += needed;
            load[c] += cost[c];
        }
    }
    Ok(bands)
}

fn tile_order(r: &Request, h: &Hardware, bands: &[Band], c: usize) -> VecDeque<(usize, usize)> {
    let local: Vec<_> = bands
        .iter()
        .enumerate()
        .filter(|(_, b)| b.core == c)
        .map(|(i, _)| i)
        .collect();
    let mut out = VecDeque::new();
    // K-major within bounded bands; weight tile serves all its M blocks before
    // retirement. This is a declared compiler schedule, not a best-schedule claim.
    for chunk in local.chunks(h.band_window) {
        let depth = chunk
            .iter()
            .map(|&b| ceil(r.jobs[bands[b].job].k, 512))
            .max()
            .unwrap();
        for kt in 0..depth {
            for &b in chunk {
                if kt < ceil(r.jobs[bands[b].job].k, 512) {
                    out.push_back((b, kt));
                }
            }
        }
    }
    out
}
fn weight_spans(
    slot: usize,
    first_k: usize,
    valid_k: usize,
    valid_n: usize,
) -> Vec<(usize, usize)> {
    (0..valid_n)
        .map(|n| (slot * 4096 + n * 1024 + first_k * 2, valid_k * 2))
        .collect()
}
fn tree(mut p: Vec<f32>) -> f32 {
    while p.len() > 1 {
        p = p.chunks_exact(2).map(|x| x[0] + x[1]).collect();
    }
    p[0]
}

pub fn run(r: &Request, h: &Hardware, values: Option<&Operands>) -> Result<Report, String> {
    validate(r, h)?;
    if let Some(v) = values {
        v.validate(r)?;
    }
    let nc = r.m_lanes.len();
    let groups = 512 / h.parallel_k;
    let dot_tail =
        h.mul_latency + log2(h.parallel_k) * h.add_latency + log2(groups) * h.add_latency;
    let mut native = if h.source == "native" {
        Some(moe_spatial::native::Native::new(h.native.clone(), r)?)
    } else {
        None
    };
    if let Some(n) = &mut native {
        n.enable_consumer_backpressure();
    }
    let ideal = h.source == "compute_oracle";
    let bands = map_bands(r, h)?;
    let ingress_bytes = h.native.outstanding_sectors * 32;
    let wslots = h
        .weight_bytes_total
        .checked_sub(ingress_bytes)
        .ok_or("ingress exceeds W budget")?
        / nc
        / 4096;
    if wslots == 0 {
        return Err("no weight slot".into());
    }
    if h.x_reuse_bands > wslots || h.x_reuse_bands > h.band_window {
        return Err("X reuse group exceeds resident weight slots or band window".into());
    }
    let mut queues: Vec<VecDeque<TilePlan>> = (0..nc)
        .map(|c| {
            group_tiles(
                tile_order(r, h, &bands, c),
                h.x_reuse_bands,
                &bands,
                &r.jobs,
                r.m_lanes[c],
                h.x_slots_per_core,
            )
        })
        .collect();
    let mut reuse_cursor = vec![ReuseCursor::default(); nc];
    let mut free_w: Vec<VecDeque<usize>> = (0..nc).map(|_| (0..wslots).collect()).collect();
    let mut live: Vec<VecDeque<usize>> = vec![VecDeque::new(); nc];
    let mut tiles = Vec::<Tile>::new();
    let mut tasks = Vec::<Task>::new();
    let mut stages: Vec<VecDeque<usize>> = vec![VecDeque::new(); nc];
    let mut free_x: Vec<VecDeque<usize>> =
        (0..nc).map(|_| (0..h.x_slots_per_core).collect()).collect();
    let mut x_cache = vec![vec![None; h.x_slots_per_core]; nc];
    let mut weight_gather_cache = vec![None; nc];
    let mut contexts = vec![0; nc];
    let mut busy_feed = vec![None; nc];
    let mut arithmetic_until = vec![0_u64; nc];
    let mut last_first_feed = vec![None; nc];
    let mut wb: Vec<_> = (0..nc)
        .map(|_| {
            Banks::new(
                h.weight_banks_total / nc,
                h.bank_word_bytes,
                h.bank_read_latency,
            )
        })
        .collect();
    let mut xb: Vec<_> = r
        .m_lanes
        .iter()
        .map(|m| {
            Banks::new(
                m * h.activation_banks_per_m,
                h.bank_word_bytes,
                h.bank_read_latency,
            )
        })
        .collect();
    let mut ab: Vec<_> = r
        .m_lanes
        .iter()
        .map(|m| {
            Banks::new(
                m * h.accumulator_banks_per_m,
                h.bank_word_bytes,
                h.bank_read_latency,
            )
        })
        .collect();
    let mut committed: Vec<Vec<usize>> = bands.iter().map(|b| vec![0; r.jobs[b.job].m]).collect();
    let mut output: Option<Vec<Vec<f32>>> =
        values.map(|_| r.jobs.iter().map(|j| vec![0.; j.m * j.n]).collect());
    let mut events: BTreeMap<u64, Vec<Event>> = BTreeMap::new();
    let mut pending_source: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut reports: Vec<_> = (0..nc)
        .map(|c| CoreReport {
            accumulator_bytes: bands
                .iter()
                .filter(|b| b.core == c)
                .map(|b| r.jobs[b.job].m * 32)
                .sum::<usize>()
                + 4096 * r.m_lanes[c] / 6,
            ..Default::default()
        })
        .collect();
    let mut now = 0;
    let mut source_bytes = 0;
    let mut control_free = 0;
    let mut control_cycles = 0;
    let mut input_bus = StreamBus {
        cycle: 0,
        used: 0,
        lanes: 6 * h.activation_banks_per_m,
    };
    let mut activation_input_bytes = 0;
    let mut trace = Vec::new();
    let ctl = h.descriptor_bytes.div_ceil(h.control_word_bytes) as u64;
    // Shared X source bus: aggregate physical bank write width, one beat/clock.
    loop {
        if now > 100_000_000 {
            return Err("structural model exceeded cycle bound".into());
        }
        if let Some(src) = &mut native {
            let completed = src.advance(now)?;
            for (id, address) in src.take_sector_completions() {
                let t = &mut tiles[id];
                let b = &bands[t.band];
                let job = &r.jobs[b.job];
                let offset = address - job.expert as u64 * h.native.expert_stride_bytes;
                let stride = (job.k * 2).div_ceil(32) * 32;
                let row = offset as usize / stride;
                let kbyte = offset as usize % stride;
                let local = t.slot * 4096 + (row - b.nt * 4) * 1024 + kbyte - t.kt * 1024;
                let done = wb[b.core].access(now, &[(local, 32)], false);
                t.landing_done = t.landing_done.max(done);
                events.entry(done).or_default().push(Event::SectorAck);
            }
            for id in completed {
                tiles[id].ready = Some(tiles[id].landing_done);
            }
        }
        if let Some(es) = events.remove(&now) {
            for e in es {
                match e {
                    Event::SectorAck => native.as_mut().unwrap().acknowledge_sector(),
                    Event::Source(id) => {
                        let t = &mut tiles[id];
                        let b = &bands[t.band];
                        let job = &r.jobs[b.job];
                        let spans = weight_spans(
                            t.slot,
                            0,
                            (job.k - t.kt * 512).min(512),
                            (job.n - b.nt * 4).min(4),
                        );
                        t.ready = Some(if ideal {
                            now
                        } else {
                            wb[b.core].access(now, &spans, false)
                        });
                    }
                    Event::Feed(c, id) => {
                        let t = &mut tasks[id];
                        weight_gather_cache[c] = Some((t.tile, t.group));
                        t.group += 1;
                        let tail = if t.group == groups {
                            dot_tail
                        } else {
                            h.mul_latency + log2(h.parallel_k) * h.add_latency
                        };
                        arithmetic_until[c] = arithmetic_until[c].max(now + tail);
                        busy_feed[c] = None;
                        if t.group == groups {
                            let tile = &mut tiles[t.tile];
                            let b = &bands[tile.band];
                            tile.consumed_rows += t.rows;
                            stages[c].pop_front();
                            free_x[c].push_back(t.xslot);
                            let done = now + dot_tail;
                            events
                                .entry(done)
                                .or_default()
                                .push(Event::Dot(c, id, tile.band, tile.kt));
                            if tile.consumed_rows == r.jobs[b.job].m {
                                if h.x_reuse_bands == 0 {
                                    assert_eq!(live[c].pop_front(), Some(t.tile));
                                } else {
                                    let pos = live[c].iter().position(|&id| id == t.tile).unwrap();
                                    live[c].remove(pos);
                                }
                                free_w[c].push_back(tile.slot);
                            }
                            if h.record_trace {
                                trace.push(serde_json::json!({"event":"last_feed","cycle":now,"core":c,"task":id,"dot_done":done}));
                            }
                        }
                    }
                    Event::Dot(c, id, band, kt) => {
                        let t = &tasks[id];
                        let b = &bands[band];
                        let job = &r.jobs[b.job];
                        let n = (job.n - b.nt * 4).min(4);
                        let spans: Vec<_> = (t.row..t.row + t.rows)
                            .map(|row| (b.offset + row * 32, n * 4))
                            .collect();
                        let end = if ideal {
                            now + h.add_latency
                        } else {
                            ab[c].rmw(now, &spans, h.add_latency)
                        };
                        events
                            .entry(end)
                            .or_default()
                            .push(Event::Commit(c, id, band, kt));
                        if let (Some(vals), Some(out)) = (values, output.as_mut()) {
                            let v = &vals.jobs[b.job];
                            for row in t.row..t.row + t.rows {
                                for col in b.nt * 4..b.nt * 4 + n {
                                    let mut products = vec![0.; 512];
                                    for (kk, p) in products
                                        .iter_mut()
                                        .enumerate()
                                        .take((job.k - kt * 512).min(512))
                                    {
                                        *p = v.x[row * job.k + kt * 512 + kk]
                                            * v.w[col * job.k + kt * 512 + kk];
                                    }
                                    // Group subtrees and the final subtree preserve the same
                                    // balanced 512-leaf FP32 sum for every power-of-two folding.
                                    let partials = products
                                        .chunks(h.parallel_k)
                                        .map(|p| tree(p.to_vec()))
                                        .collect();
                                    out[b.job][row * job.n + col] += tree(partials);
                                }
                            }
                        }
                    }
                    Event::Commit(c, id, band, kt) => {
                        let t = &tasks[id];
                        for progress in committed[band].iter_mut().skip(t.row).take(t.rows) {
                            assert_eq!(*progress, kt);
                            *progress += 1;
                        }
                        contexts[c] -= 1;
                        reports[c].done_cycle = now;
                        if h.record_trace {
                            trace.push(serde_json::json!({"event":"commit","cycle":now,"core":c,"task":id}));
                        }
                    }
                }
            }
        }
        if queues.iter().all(|q| q.is_empty())
            && live.iter().all(|q| q.is_empty())
            && stages.iter().all(|q| q.is_empty())
            && contexts.iter().all(|&v| v == 0)
            && events.is_empty()
            && native.as_ref().is_none_or(|n| n.drained())
        {
            break;
        }

        for c in 0..nc {
            // One descriptor sequencer per shared port; finite reservation before
            // source submission. Descriptor service is derived from words, not 2/3/2.
            if !queues[c].is_empty() && !free_w[c].is_empty() && (ideal || now >= control_free) {
                let TilePlan {
                    band,
                    kt,
                    sequence,
                    group_len,
                } = queues[c].pop_front().unwrap();
                let b = &bands[band];
                let j = &r.jobs[b.job];
                let slot = free_w[c].pop_front().unwrap();
                let id = tiles.len();
                let release = if ideal { now } else { now + ctl };
                if !ideal {
                    control_free = release;
                    control_cycles += ctl;
                }
                tiles.push(Tile {
                    band,
                    kt,
                    slot,
                    ready: None,
                    row: 0,
                    consumed_rows: 0,
                    landing_done: 0,
                    sequence,
                    group_len,
                });
                live[c].push_back(id);
                source_bytes +=
                    ((j.n - b.nt * 4).min(4) * ceil((j.k - kt * 512).min(512) * 2, 32) * 32) as u64;
                // Source descriptors must not become visible before their service
                // ends. Requests are sent at release via the pending list below.
                pending_source.entry(release).or_default().push(id);
            }
        }
        if let Some(ids) = pending_source.remove(&now) {
            for id in ids {
                let t = &tiles[id];
                let b = &bands[t.band];
                let job = &r.jobs[b.job];
                if let Some(src) = &mut native {
                    src.submit(WeightRead {
                        id,
                        job,
                        n_start: b.nt * 4,
                        k_start: t.kt * 512,
                        valid_n: (job.n - b.nt * 4).min(4),
                        valid_k: (job.k - t.kt * 512).min(512),
                        now,
                    })?;
                } else {
                    events.entry(now + 1).or_default().push(Event::Source(id));
                }
            }
        }

        for c in 0..nc {
            // Static loop interchange, not readiness-based out-of-order issue.
            // A bounded tag lookup resolves the sequencer's one target tile;
            // at most one task is prepared per cycle, as in the frozen policy.
            let mut prepare_wait = "output_dependency";
            let next_tile = if h.x_reuse_bands == 0 {
                live[c].front().copied()
            } else if live[c].front().is_some_and(|&id| {
                tiles[id].group_len == 1 && tiles[id].sequence < reuse_cursor[c].target_sequence()
            }) {
                // A one-tile group uses the frozen retirement boundary. In
                // particular, a cache-fitting expert must not acquire an
                // unrelated next-tile staging optimization through this mode.
                None
            } else {
                live[c]
                    .iter()
                    .copied()
                    .find(|&id| tiles[id].sequence == reuse_cursor[c].target_sequence())
            };
            if h.x_reuse_bands > 0 && next_tile.is_none() && !queues[c].is_empty() {
                prepare_wait = "target_descriptor_not_admitted";
            }
            if let (Some(tile_id), Some(&fallback_slot)) = (next_tile, free_x[c].front()) {
                let t = &mut tiles[tile_id];
                let b = &bands[t.band];
                let j = &r.jobs[b.job];
                if t.row < j.m && committed[t.band][t.row] == t.kt {
                    let row = t.row;
                    let rows = (j.m - row).min(r.m_lanes[c]);
                    if (row..row + rows).all(|rr| committed[t.band][rr] == t.kt) {
                        let vk = (j.k - t.kt * 512).min(512);
                        let bytes = rows * vk * 2;
                        let key = (b.job, row, rows, t.kt);
                        // If this exact X is still feeding the array, retain it
                        // and wait for that slot; do not duplicate it into the
                        // other stage. The slot is reusable only after last_feed.
                        let busy_match = h.x_reuse_bands > 0
                            && x_cache[c]
                                .iter()
                                .enumerate()
                                .any(|(slot, tag)| *tag == Some(key) && !free_x[c].contains(&slot));
                        if busy_match {
                            reports[c].x_reuse_busy_tag_cycles += 1;
                            prepare_wait = "x_reuse_slot_busy";
                        } else {
                            let xslot = free_x[c]
                                .iter()
                                .copied()
                                .find(|&slot| x_cache[c][slot] == Some(key))
                                .unwrap_or(fallback_slot);
                            let cached = x_cache[c][xslot] == Some(key);
                            if cached {
                                reports[c].x_cache_hits += 1;
                            } else {
                                reports[c].x_cache_misses += 1;
                                reports[c].x_input_bytes += bytes as u64;
                            }
                            let spans: Vec<_> = (0..rows)
                                .map(|rr| ((xslot * r.m_lanes[c] + rr) * 1024, vk * 2))
                                .collect();
                            let ready = if ideal || cached {
                                now
                            } else {
                                xb[c].stream_write(now, &spans, &mut input_bus)
                            };
                            let id = tasks.len();
                            tasks.push(Task {
                                tile: tile_id,
                                row,
                                rows,
                                xslot,
                                x_ready: ready,
                                group: 0,
                                feed_ready: 0,
                                started: false,
                            });
                            t.row += rows;
                            if h.x_reuse_bands > 0 {
                                assert_eq!(row, reuse_cursor[c].row);
                                reuse_cursor[c].advance(t.group_len, rows, j.m);
                            }
                            let pos = free_x[c].iter().position(|&slot| slot == xslot).unwrap();
                            free_x[c].remove(pos);
                            x_cache[c][xslot] = Some(key);
                            stages[c].push_back(id);
                            if !cached {
                                activation_input_bytes += bytes as u64;
                            }
                        }
                    }
                }
            }
            let mut state = "idle";
            if busy_feed[c].is_some() {
                state = "bank_read_in_flight";
            } else if let Some(&id) = stages[c].front() {
                let t = &mut tasks[id];
                let tile = &tiles[t.tile];
                let b = &bands[tile.band];
                let j = &r.jobs[b.job];
                if tile.ready.is_none_or(|v| v > now) {
                    state = "weight_not_ready";
                    if tile.ready.is_none() {
                        reports[c].weight_transport_wait_cycles += 1;
                    } else {
                        reports[c].weight_landing_wait_cycles += 1;
                    }
                } else if t.x_ready > now {
                    state = "activation_not_ready";
                } else if !t.started && contexts[c] >= h.result_contexts_per_core {
                    state = "result_context_full";
                } else if now < t.feed_ready {
                    state = "multiplier_issue";
                } else {
                    if !t.started {
                        t.started = true;
                        contexts[c] += 1;
                        reports[c].result_context_peak =
                            reports[c].result_context_peak.max(contexts[c]);
                        reports[c].invocations += 1;
                        if let Some(previous) = last_first_feed[c] {
                            let gap = now - previous;
                            reports[c].first_feed_gap_min =
                                Some(reports[c].first_feed_gap_min.map_or(gap, |v| v.min(gap)));
                            reports[c].first_feed_gap_max = reports[c].first_feed_gap_max.max(gap);
                            reports[c].first_feed_gap_sum += gap;
                        }
                        last_first_feed[c] = Some(now);
                        let vk = (j.k - tile.kt * 512).min(512);
                        let vn = (j.n - b.nt * 4).min(4);
                        reports[c].useful_macs += (t.rows * vn * vk) as u64;
                        reports[c].padded_macs += (r.m_lanes[c] * 4 * 512) as u64;
                        if h.record_trace {
                            trace.push(serde_json::json!({"event":"first_feed","cycle":now,"core":c,"task":id,"job":b.job,"nt":b.nt,"kt":tile.kt,"row":t.row,"rows":t.rows}));
                        }
                    }
                    let kk = t.group * h.parallel_k;
                    let vk = (j.k - tile.kt * 512)
                        .min(512)
                        .saturating_sub(kk)
                        .min(h.parallel_k);
                    let vn = (j.n - b.nt * 4).min(4);
                    let ws = weight_spans(tile.slot, kk, vk, vn);
                    let xs: Vec<_> = (0..t.rows)
                        .map(|rr| ((t.xslot * r.m_lanes[c] + rr) * 1024 + kk * 2, vk * 2))
                        .collect();
                    let done = if ideal {
                        now + 1
                    } else {
                        (if weight_gather_cache[c] == Some((t.tile, t.group)) {
                            now
                        } else {
                            wb[c].access(now, &ws, true)
                        })
                        .max(xb[c].access(now, &xs, true))
                        .max(now + 1)
                    };
                    busy_feed[c] = Some(id);
                    t.feed_ready = done;
                    reports[c].feed_groups += 1;
                    events.entry(done).or_default().push(Event::Feed(c, id));
                    state = "feed_started";
                }
            } else if !live[c].is_empty() {
                state = prepare_wait;
            }
            *reports[c].issue_states.entry(state.into()).or_default() += 1;
            if now < arithmetic_until[c] {
                reports[c].arithmetic_active_cycles += 1;
            }
            reports[c].weight_peak_bytes = reports[c]
                .weight_peak_bytes
                .max((wslots - free_w[c].len()) * 4096);
            reports[c].activation_peak_bytes = reports[c]
                .activation_peak_bytes
                .max((h.x_slots_per_core - free_x[c].len()) * r.m_lanes[c] * 1024);
        }
        now += 1;
    }
    let mut exact = None;
    if let (Some(v), Some(out)) = (values, &output) {
        for (j, o) in out.iter().enumerate() {
            let reference = v.reference(r, j);
            if o.iter()
                .map(|x| x.to_bits())
                .ne(reference.iter().map(|x| x.to_bits()))
            {
                return Err(format!("numerical mismatch job {j}"));
            }
        }
        exact = Some(true);
    }
    for (b, k) in bands.iter().zip(&committed) {
        assert!(k.iter().all(|&v| v == ceil(r.jobs[b.job].k, 512)));
    }
    assert!(free_w.iter().all(|s| s.len() == wslots));
    assert!(free_x.iter().all(|s| s.len() == h.x_slots_per_core));
    let macs = reports.iter().map(|c| c.useful_macs).sum();
    assert_eq!(
        macs,
        r.jobs.iter().map(|j| (j.m * j.n * j.k) as u64).sum::<u64>()
    );
    let padded = reports.iter().map(|c| c.padded_macs).sum();
    let native_report = native.as_mut().map(|n| n.report());
    if let Some(n) = &native_report {
        assert_eq!(
            n["counts"]["native_read_bytes"].as_u64(),
            Some(source_bytes)
        );
    }
    let bill = serde_json::json!({
        "logical_tile_m":r.m_lanes,"logical_tile_n":4,"logical_tile_k":512,
        "physical_multipliers_total":6*4*h.parallel_k,"physical_parallel_k":h.parallel_k,
        "serial_k_groups":groups,"minimum_feed_cycles_per_tile":groups,
        "tail_after_last_feed_cycles":dot_tail,"latencies_are_assumptions_not_synthesis":true,
        "weight_sram_bytes":h.weight_bytes_total,"x_sram_bytes":h.activation_bytes_total,
        "shared_ingress_bytes_charged_inside_weight_budget":ingress_bytes,
        "private_weight_slots_per_core":wslots,
        "private_weight_payload_bytes_total":nc*wslots*4096,
        "accumulator_budget_bytes":h.accumulator_bytes_total,"control_reservation_bytes":4096,
        "intra_group_adders_total":24*(h.parallel_k-1),"cross_group_adders_total":24*(groups-1),
        "arithmetic_pipeline_register_bytes_lower_bound":24*4*(h.parallel_k*h.mul_latency as usize+(h.parallel_k-1)*h.add_latency as usize),
        "group_context_register_bytes":24*groups*4*h.result_contexts_per_core,
        "operand_gather_register_bytes":(6+nc*4)*h.parallel_k*2,
        "activation_source_bus_register_bytes":6*h.activation_banks_per_m*h.bank_word_bytes,
        "cross_group_pipeline_register_bytes_lower_bound":24*4*(groups-1)*h.add_latency as usize,
        "bank_response_latches_bytes":h.bank_word_bytes*(h.weight_banks_total+6*h.activation_banks_per_m)*h.bank_read_latency as usize,
        "physical_weight_banks":h.weight_banks_total,"physical_x_banks":6*h.activation_banks_per_m,
        "physical_accumulator_banks":6*h.accumulator_banks_per_m,
        "fp32_accumulator_adders":6*h.accumulator_banks_per_m*h.bank_word_bytes/4,
        "controller_state_bytes":nc*(h.band_window*16+h.result_contexts_per_core*32+h.x_slots_per_core*32+64)+h.weight_bytes_total/4096*32+64+reuse_state_bytes(r,h),
        "x_reuse_added_control_bytes":reuse_state_bytes(r,h),
        "x_reuse_group_bands":h.x_reuse_bands,
        "x_reuse_lookup":"one target sequence tag; bounded W-slot comparators; one sequencer transition/core/cycle; no ready-window search",
        "bank_ports":"1RW; read/write share a command port; atomic RMW bank lock",
        "bank_address":"word_address modulo private bank count; W slot-major, X stage-major, output band-major",
        "mapping":"offline deterministic; bounded K-major band window; whole N-band ownership; optional resident-group M/N loop interchange",
        "equal_mac_count_not_equal_area":true});
    Ok(Report{schema:"plena_structural_analytic_v1".into(),scope:"resource-derived candidate model; six GEMMs when summed; NOT hardware measured or model E2E".into(),
        source:h.source.clone(),hardware:h.clone(),cycles:now,time_ms_at_assumed_clock:now as f64*h.core_period_ps as f64/1e9,
        resource_bill:bill,cores:reports,weight_banks:wb,activation_banks:xb,accumulator_banks:ab,
        source_bytes,control_cycles,activation_input_bytes,useful_macs:macs,padded_macs:padded,numerical_bit_exact:exact,
        output_fp32_bits:output.map(|v|v.iter().map(|x|x.iter().map(|f|f.to_bits()).collect()).collect()),
        drained:true,ownership_and_k_order_verified:true,trace,native:native_report})
}

#[cfg(test)]
mod tests;
