//! Finite, causally gated operand fabric. See doc/moe_spatial_fabric_contract.md.
use super::*;
use sha2::{Digest, Sha256};
mod private_memory;
use private_memory::PrivateState;
pub use private_memory::{PrivateConfig, PrivateReport};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Selector {
    Fifo,
    Affinity,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    Invocation,
    Cohort,
    TileCohort,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fabric {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_memories: Option<PrivateConfig>,
    pub weight_bpc: u64,
    pub activation_bpc: u64,
    pub accumulator_bpc: u64,
    pub control_ports: usize,
    pub issue_cycles: u64,
    pub install_cycles: u64,
    pub completion_cycles: u64,
    pub broadcast: bool,
    pub retain_weights: bool,
    pub hop_cycles: u64,
    pub slots_per_m: usize,
    pub stages_per_core: usize,
    pub descriptors: usize,
    pub ready_window: usize,
    pub accumulator_bytes: usize,
    pub packed_operand_ports: bool,
    pub selector: Selector,
    pub control: Control,
    pub zero_control_time: bool,
    pub zero_weight_time: bool,
    pub zero_activation_time: bool,
    pub zero_accumulator_time: bool,
}
impl Default for Fabric {
    fn default() -> Self {
        Self {
            private_memories: None,
            weight_bpc: 1024,
            activation_bpc: 6144,
            accumulator_bpc: 192,
            control_ports: 1,
            issue_cycles: 2,
            install_cycles: 3,
            completion_cycles: 2,
            broadcast: true,
            retain_weights: true,
            hop_cycles: 1,
            slots_per_m: 2,
            stages_per_core: 2,
            descriptors: 256,
            ready_window: 32,
            accumulator_bytes: 2 * 1024 * 1024,
            packed_operand_ports: false,
            selector: Selector::Affinity,
            control: Control::Cohort,
            zero_control_time: false,
            zero_weight_time: false,
            zero_activation_time: false,
            zero_accumulator_time: false,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub compute: Request,
    pub fabric: Fabric,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Service {
    pub resource: String,
    pub port: usize,
    pub released: u64,
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
    pub requests: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_start: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_end: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_rate: Option<u64>,
}
struct Port {
    free: Vec<u64>,
    name: &'static str,
    port_base: usize,
    byte_cursor: u64,
}
impl Port {
    fn new(name: &'static str, count: usize) -> Self {
        Self {
            free: vec![0; count],
            name,
            port_base: 0,
            byte_cursor: 0,
        }
    }
    fn reserve(
        &mut self,
        now: u64,
        cycles: u64,
        bytes: u64,
        ids: &[usize],
        log: &mut Vec<Service>,
    ) -> u64 {
        let port = (0..self.free.len())
            .min_by_key(|&p| (self.free[p], p))
            .unwrap();
        let start = now.max(self.free[port]);
        let end = start + cycles;
        self.free[port] = end;
        log.push(Service {
            resource: self.name.into(),
            port: port + self.port_base,
            released: now,
            start,
            end,
            bytes,
            requests: ids.to_vec(),
            byte_start: None,
            byte_end: None,
            byte_rate: None,
        });
        end
    }
    fn reserve_data(
        &mut self,
        now: u64,
        bytes: u64,
        timing: (u64, bool, bool),
        ids: &[usize],
        log: &mut Vec<Service>,
    ) -> u64 {
        let (rate, zero, packed) = timing;
        if !packed || zero {
            return self.reserve(now, service(bytes, rate, zero), bytes, ids, log);
        }
        let begin = self.byte_cursor.max(now * rate);
        let finish = begin + bytes;
        self.byte_cursor = finish;
        let start = begin / rate;
        let end = finish.div_ceil(rate);
        self.free[0] = end;
        log.push(Service {
            resource: self.name.into(),
            port: self.port_base,
            released: now,
            start,
            end,
            bytes,
            requests: ids.to_vec(),
            byte_start: Some(begin),
            byte_end: Some(finish),
            byte_rate: Some(rate),
        });
        end
    }
}
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Stats {
    pub source_weight_bytes: u64,
    pub delivered_weight_bytes: u64,
    pub activation_bytes: u64,
    pub accumulator_rmw_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub broadcast_transfers: u64,
    pub issue_services: u64,
    pub install_services: u64,
    pub completion_services: u64,
    pub descriptor_peak: usize,
    pub cohort_descriptor_peak: usize,
    pub total_metadata_peak: usize,
    pub resume_selections: usize,
    pub weight_storage_peak_bytes: usize,
    pub activation_storage_peak_bytes: usize,
    pub result_storage_peak_bytes: usize,
    pub simultaneous_operand_result_peak_bytes: usize,
    pub weight_budget_bytes: usize,
    pub activation_budget_bytes: usize,
    pub result_budget_bytes: usize,
    pub output_storage_bytes: usize,
    pub control_service_cycles: u64,
    pub weight_service_cycles: u64,
    pub activation_service_cycles: u64,
    pub accumulator_service_cycles: u64,
}
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Core {
    pub m_lanes: usize,
    pub invocations: u64,
    pub useful_macs: u64,
    pub issued_mac_slots: u64,
    pub pipeline_peak: usize,
    pub stage_peak: usize,
    pub states: BTreeMap<String, u64>,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Trace {
    pub request: usize,
    pub group: usize,
    pub core: usize,
    pub expert: usize,
    pub rows: Vec<usize>,
    pub n_start: usize,
    pub k_start: usize,
    pub valid_n: usize,
    pub valid_k: usize,
    pub admitted: u64,
    pub descriptor_ready: u64,
    pub weight_ready: u64,
    pub activation_ready: u64,
    pub issue_cycle: u64,
    pub mac_done: u64,
    pub rmw_done: u64,
    pub commit_cycle: u64,
    pub useful_macs: u64,
    pub issued_mac_slots: u64,
    pub slot: usize,
    pub cache_hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operand_ready: Option<u64>,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Output {
    pub name: String,
    pub scope: String,
    pub total_cycles: u64,
    pub total_multipliers: usize,
    pub useful_macs: u64,
    pub issued_mac_slots: u64,
    pub numerical_bit_exact: Option<bool>,
    pub output_fp32_bits: Option<Vec<Vec<u32>>>,
    pub output_bf16_bits: Option<Vec<Vec<u16>>>,
    pub drained: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_memories: Option<PrivateReport>,
    pub stats: Stats,
    pub cores: Vec<Core>,
    pub trace: Vec<Trace>,
    pub services: Vec<Service>,
    pub invocation_audit_count: usize,
    pub service_audit_count: usize,
    pub invocation_sha256: String,
    pub service_sha256: String,
}
#[derive(Default)]
struct Slot {
    key: Option<Key>,
    ready: u64,
    refs: usize,
    touched: u64,
    values: Vec<f32>,
}
struct Task {
    key: Key,
    trace: Trace,
    values: Vec<f32>,
    activation: Option<Vec<f32>>,
    miss: bool,
}
struct Group {
    ids: Vec<usize>,
    rmw_left: usize,
    rows_left: usize,
    header_ready: u64,
    local_ready: u64,
}
struct Transfer {
    key: Key,
    targets: Vec<(usize, usize, usize)>,
    end: u64,
    bytes: u64,
    service: usize,
}
enum Event {
    Dispatch(Vec<usize>),
    ActivationArrive(usize, u64),
    ActivationFill(usize),
    StoreWeight(Key, Vec<(usize, usize, usize)>),
    ReadDone(usize),
    Arrive(Key, Vec<(usize, usize, usize)>),
    Fill(Key, Vec<(usize, usize, usize)>),
    MacDone(usize),
    RmwDone(usize),
    Commit(Vec<usize>),
    FinishGroup(usize),
    RetireGroup(usize),
}

fn aligned(bytes: usize) -> u64 {
    bytes.div_ceil(32) as u64 * 32
}
fn service(bytes: u64, bpc: u64, zero: bool) -> u64 {
    if zero { 0 } else { bytes.div_ceil(bpc) }
}
fn control(cycles: u64, f: &Fabric) -> u64 {
    if f.zero_control_time { 0 } else { cycles }
}
fn weight_values(r: &Request, key: Key, external: Option<&operands::Operands>) -> Vec<f32> {
    let j = &r.jobs[key.job];
    let mut v = vec![0.; r.n_lanes * r.k_lanes];
    for n in 0..r.n_lanes.min(j.n - key.nt * r.n_lanes) {
        for k in 0..r.k_lanes.min(j.k - key.kt * r.k_lanes) {
            let col = key.nt * r.n_lanes + n;
            let kk = key.kt * r.k_lanes + k;
            v[n * r.k_lanes + k] = external.map_or_else(
                || w_value(j, kk, col),
                |d| d.jobs[key.job].w[col * j.k + kk],
            );
        }
    }
    v
}
fn multiply(
    r: &Request,
    t: &Task,
    weights: &[f32],
    external: Option<&operands::Operands>,
) -> Vec<f32> {
    let j = &r.jobs[t.key.job];
    let mut v = Vec::new();
    for (ri, &row) in t.trace.rows.iter().enumerate() {
        for n in 0..t.trace.valid_n {
            let mut p = vec![0.; r.k_lanes];
            for k in 0..t.trace.valid_k {
                let kk = t.trace.k_start + k;
                let x = t.activation.as_ref().map_or_else(
                    || {
                        external.map_or_else(
                            || x_value(j, row, kk),
                            |d| d.jobs[t.key.job].x[row * j.k + kk],
                        )
                    },
                    |x| x[ri * r.k_lanes + k],
                );
                p[k] = x * weights[n * r.k_lanes + k];
            }
            v.push(tree_reduce(p));
        }
    }
    v
}

pub fn run(input: &Input) -> Result<Output, String> {
    run_with_operands(input, None)
}

/// One finite 2D weight DMA descriptor. K offsets/lengths are element counts.
pub struct WeightRead<'a> {
    pub id: usize,
    pub job: &'a Job,
    pub n_start: usize,
    pub k_start: usize,
    pub valid_n: usize,
    pub valid_k: usize,
    pub now: u64,
}
/// A live source: completion, not a precomputed trace, gates downstream delivery.
pub trait WeightSource {
    fn submit(&mut self, request: WeightRead<'_>) -> Result<(), String>;
    fn advance(&mut self, now: u64) -> Result<Vec<usize>, String>;
    fn drained(&self) -> bool;
}

pub fn run_with_operands(
    input: &Input,
    external: Option<&operands::Operands>,
) -> Result<Output, String> {
    run_with_source(input, external, None)
}

pub fn run_with_source(
    input: &Input,
    external: Option<&operands::Operands>,
    mut source: Option<&mut dyn WeightSource>,
) -> Result<Output, String> {
    let (r, f) = (&input.compute, &input.fabric);
    validate(r)?;
    if let Some(d) = external {
        d.validate(r)?;
    }
    if f.weight_bpc == 0
        || f.activation_bpc == 0
        || f.accumulator_bpc == 0
        || f.control_ports == 0
        || f.control_ports > 8
        || f.slots_per_m == 0
        || f.slots_per_m > 16
        || f.stages_per_core == 0
        || f.stages_per_core > 8
        || f.descriptors == 0
        || f.descriptors > 4096
        || f.ready_window == 0
        || f.weight_bpc > (1 << 30)
        || f.activation_bpc > (1 << 30)
        || f.accumulator_bpc > (1 << 30)
        || (f.control == Control::TileCohort && f.descriptors < 2)
        || r.jobs.iter().any(|j| j.expert > u32::MAX as usize)
    {
        return Err("invalid finite fabric budget".into());
    }
    let output_bytes: usize = r.jobs.iter().map(|j| j.m * j.n * 4).sum();
    if output_bytes > f.accumulator_bytes {
        return Err("output accumulator capacity exceeded".into());
    }
    let nc = r.m_lanes.len();
    let cap = r.result_latency_cycles.div_ceil(r.issue_interval_cycles) as usize;
    let weight_size = r.n_lanes * r.k_lanes * 2;
    let mut stats = Stats {
        weight_budget_bytes: r.m_lanes.iter().sum::<usize>() * f.slots_per_m * weight_size,
        activation_budget_bytes: r.m_lanes.iter().sum::<usize>()
            * f.stages_per_core
            * r.k_lanes
            * 2,
        result_budget_bytes: r.m_lanes.iter().sum::<usize>() * cap * r.n_lanes * 4,
        output_storage_bytes: output_bytes,
        ..Default::default()
    };
    let mut cores: Vec<Core> = r
        .m_lanes
        .iter()
        .map(|&m| Core {
            m_lanes: m,
            ..Default::default()
        })
        .collect();
    let owner = owners(r);
    let mut private = f
        .private_memories
        .as_ref()
        .map(|cfg| PrivateState::new(cfg.clone(), r, f, &owner))
        .transpose()?;
    let mut ready = Ready::default();
    let mut progress = Vec::new();
    let mut in_use = Vec::new();
    let mut remaining = vec![0usize; nc];
    for (i, j) in r.jobs.iter().enumerate() {
        let count = j.m * j.n.div_ceil(r.n_lanes);
        progress.push(vec![0; count]);
        in_use.push(vec![false; count]);
        remaining[owner[i]] += count;
        for nt in 0..j.n.div_ceil(r.n_lanes) {
            for row in 0..j.m {
                ready.add(Key { job: i, nt, kt: 0 }, row);
            }
        }
    }
    let mut values = (r.verify_values && private.is_none()).then(|| {
        r.jobs
            .iter()
            .map(|j| vec![0f32; j.m * j.n])
            .collect::<Vec<_>>()
    });
    let mut slots: Vec<Vec<Slot>> = r
        .m_lanes
        .iter()
        .enumerate()
        .map(|(c, m)| {
            (0..private
                .as_ref()
                .map_or(m * f.slots_per_m, |p| p.config.weight_slots[c]))
                .map(|_| Slot::default())
                .collect()
        })
        .collect();
    let mut stages: Vec<VecDeque<usize>> = vec![VecDeque::new(); nc];
    let mut pending = vec![0usize; nc];
    let mut next_issue = vec![0u64; nc];
    let mut local_commit_free = vec![0u64; nc];
    let mut tasks: Vec<Task> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    let mut persistent: BTreeMap<Key, usize> = BTreeMap::new();
    let mut open_cohorts = 0usize;
    let mut transfers: Vec<Transfer> = Vec::new();
    let mut flights: BTreeMap<Key, usize> = BTreeMap::new();
    let mut events: BTreeMap<u64, Vec<Event>> = BTreeMap::new();
    let mut cp = Port::new("control", f.control_ports);
    let mut wp = Port::new("weight", 1);
    let mut xp = Port::new("activation", 1);
    let mut ap = Port::new("accumulator", 1);
    let mut log = Vec::new();
    let mut active = 0usize;
    let mut now = 0u64;
    let mut cursor = 0usize;
    loop {
        if let Some(src) = source.as_deref_mut() {
            for tid in src.advance(now)? {
                // Native completion precedes the finite SRAM-facing delivery port.
                let tr = &mut transfers[tid];
                let ids: Vec<_> = tr.targets.iter().map(|x| x.2).collect();
                tr.service = log.len();
                tr.end = wp.reserve(
                    now,
                    service(tr.bytes, f.weight_bpc, f.zero_weight_time),
                    tr.bytes,
                    &ids,
                    &mut log,
                );
                events.entry(tr.end).or_default().push(Event::ReadDone(tid));
            }
        }
        // Drain current-time events twice: before admission, and after zero-time
        // admission service. Event requests reserve ports only when released.
        for phase in 0..2 {
            while let Some(es) = events.remove(&now) {
                for event in es {
                    match event {
                        Event::Dispatch(ids) => {
                            let mut misses: Vec<(usize, usize, usize)> = Vec::new();
                            for &id in &ids {
                                let t = &mut tasks[id];
                                t.trace.descriptor_ready = now;
                                let bytes = aligned(t.trace.rows.len() * t.trace.valid_k * 2);
                                stats.activation_bytes += bytes;
                                let activation_arrival = xp.reserve_data(
                                    now,
                                    bytes,
                                    (
                                        f.activation_bpc,
                                        f.zero_activation_time,
                                        f.packed_operand_ports,
                                    ),
                                    &[id],
                                    &mut log,
                                );
                                if private.is_some() {
                                    events
                                        .entry(activation_arrival)
                                        .or_default()
                                        .push(Event::ActivationArrive(id, bytes));
                                } else {
                                    t.trace.activation_ready = activation_arrival;
                                }
                                if t.miss {
                                    misses.push((t.trace.core, t.trace.slot, id));
                                }
                            }
                            let bundles = if f.broadcast {
                                vec![misses]
                            } else {
                                misses.into_iter().map(|m| vec![m]).collect()
                            };
                            for targets in bundles.into_iter().filter(|x| !x.is_empty()) {
                                let id = targets[0].2;
                                let key = tasks[id].key;
                                let bytes =
                                    aligned(tasks[id].trace.valid_n * tasks[id].trace.valid_k * 2);
                                let ids: Vec<_> = targets.iter().map(|x| x.2).collect();
                                if let Some(&tid) = flights.get(&key).filter(|_| f.broadcast)
                                    && transfers[tid].end > now
                                {
                                    if transfers[tid].service != usize::MAX {
                                        log[transfers[tid].service].requests.extend(ids);
                                    }
                                    transfers[tid].targets.extend(targets);
                                    continue;
                                }
                                let tid = transfers.len();
                                let (service_index, end) = if let Some(src) = source.as_deref_mut()
                                {
                                    let t = &tasks[id].trace;
                                    src.submit(WeightRead {
                                        id: tid,
                                        job: &r.jobs[key.job],
                                        n_start: t.n_start,
                                        k_start: t.k_start,
                                        valid_n: t.valid_n,
                                        valid_k: t.valid_k,
                                        now,
                                    })?;
                                    (usize::MAX, u64::MAX)
                                } else {
                                    let index = log.len();
                                    let end = wp.reserve(
                                        now,
                                        service(bytes, f.weight_bpc, f.zero_weight_time),
                                        bytes,
                                        &ids,
                                        &mut log,
                                    );
                                    (index, end)
                                };
                                stats.source_weight_bytes += bytes;
                                transfers.push(Transfer {
                                    key,
                                    targets,
                                    end,
                                    bytes,
                                    service: service_index,
                                });
                                if f.broadcast {
                                    flights.insert(key, tid);
                                }
                                if end != u64::MAX {
                                    events.entry(end).or_default().push(Event::ReadDone(tid));
                                }
                            }
                        }
                        Event::ActivationArrive(id, bytes) => {
                            let p = private.as_mut().unwrap();
                            let c = tasks[id].trace.core;
                            let end = p.activation_write[c].reserve_data(
                                now,
                                bytes,
                                (
                                    p.config.activation_write_bpc[c],
                                    f.zero_activation_time,
                                    false,
                                ),
                                &[id],
                                &mut log,
                            );
                            tasks[id].trace.activation_ready = end;
                            events
                                .entry(end)
                                .or_default()
                                .push(Event::ActivationFill(id));
                        }
                        Event::ActivationFill(id) => {
                            if r.verify_values {
                                let t = &mut tasks[id];
                                let j = &r.jobs[t.key.job];
                                let mut x = vec![0.; t.trace.rows.len() * r.k_lanes];
                                for (ri, &row) in t.trace.rows.iter().enumerate() {
                                    for k in 0..t.trace.valid_k {
                                        let kk = t.trace.k_start + k;
                                        x[ri * r.k_lanes + k] = external.map_or_else(
                                            || x_value(j, row, kk),
                                            |d| d.jobs[t.key.job].x[row * j.k + kk],
                                        );
                                    }
                                }
                                t.activation = Some(x);
                            }
                        }
                        Event::ReadDone(tid) => {
                            let tr = &transfers[tid];
                            if flights.get(&tr.key) == Some(&tid) {
                                flights.remove(&tr.key);
                            }
                            stats.delivered_weight_bytes += tr.bytes * tr.targets.len() as u64;
                            stats.broadcast_transfers += u64::from(tr.targets.len() > 1);
                            let hops = if tr.targets.len() > 1 {
                                (usize::BITS - (tr.targets.len() - 1).leading_zeros()) as u64
                            } else {
                                0
                            };
                            let at = now
                                + if f.zero_weight_time {
                                    0
                                } else {
                                    hops * f.hop_cycles
                                };
                            events.entry(at).or_default().push(if private.is_some() {
                                Event::StoreWeight(tr.key, tr.targets.clone())
                            } else {
                                Event::Arrive(tr.key, tr.targets.clone())
                            });
                        }
                        Event::StoreWeight(key, targets) => {
                            let mut end = now;
                            if let Some(p) = &mut private {
                                for &(c, _, id) in &targets {
                                    let t = &tasks[id].trace;
                                    let bytes = aligned(t.valid_n * t.valid_k * 2);
                                    end = end.max(p.weight_write[c].reserve_data(
                                        now,
                                        bytes,
                                        (p.config.weight_write_bpc[c], f.zero_weight_time, false),
                                        &[id],
                                        &mut log,
                                    ));
                                }
                            }
                            events
                                .entry(end)
                                .or_default()
                                .push(Event::Arrive(key, targets));
                        }
                        Event::Arrive(key, targets) => {
                            let bundles = if f.control != Control::Invocation {
                                vec![targets]
                            } else {
                                targets.into_iter().map(|t| vec![t]).collect()
                            };
                            for targets in bundles {
                                let ids: Vec<_> = targets.iter().map(|t| t.2).collect();
                                let end = cp.reserve(
                                    now,
                                    control(f.install_cycles, f),
                                    0,
                                    &ids,
                                    &mut log,
                                );
                                stats.install_services += 1;
                                events
                                    .entry(end)
                                    .or_default()
                                    .push(Event::Fill(key, targets));
                            }
                        }
                        Event::Fill(key, targets) => {
                            for (c, s, _) in targets {
                                let slot = &mut slots[c][s];
                                assert_eq!(slot.key, Some(key));
                                assert!(slot.refs > 0);
                                slot.ready = now;
                                if r.verify_values {
                                    slot.values = weight_values(r, key, external);
                                }
                            }
                        }
                        Event::MacDone(id) => {
                            let t = &tasks[id];
                            let bytes = aligned(t.trace.rows.len() * t.trace.valid_n * 8);
                            stats.accumulator_rmw_bytes += bytes;
                            let (port, rate) = match private.as_mut() {
                                Some(p) => (
                                    &mut p.accumulator[t.trace.core],
                                    p.config.accumulator_bpc[t.trace.core],
                                ),
                                None => (&mut ap, f.accumulator_bpc),
                            };
                            let end = port.reserve_data(
                                now,
                                bytes,
                                (rate, f.zero_accumulator_time, f.packed_operand_ports),
                                &[id],
                                &mut log,
                            );
                            events.entry(end).or_default().push(Event::RmwDone(id));
                        }
                        Event::RmwDone(id) => {
                            let t = &mut tasks[id];
                            t.trace.rmw_done = now;
                            let j = &r.jobs[t.key.job];
                            for (ri, &row) in t.trace.rows.iter().enumerate() {
                                assert_eq!(progress[t.key.job][t.key.nt * j.m + row], t.key.kt);
                                if let Some(p) = &mut private {
                                    p.accumulate(r, t, ri, row);
                                }
                                if let Some(ref mut out) = values {
                                    for n in 0..t.trace.valid_n {
                                        out[t.key.job][row * j.n + t.trace.n_start + n] +=
                                            t.values[ri * t.trace.valid_n + n];
                                    }
                                }
                            }
                            t.values.clear();
                            let g = t.trace.group;
                            if f.control == Control::TileCohort {
                                groups[g].rows_left -= t.trace.rows.len();
                                // One-cycle local sequencer update, not a new
                                // global descriptor visit for each M block.
                                let visible = now.max(local_commit_free[t.trace.core]) + 1;
                                local_commit_free[t.trace.core] = visible;
                                groups[g].local_ready = groups[g].local_ready.max(visible);
                                events
                                    .entry(visible)
                                    .or_default()
                                    .push(Event::Commit(vec![id]));
                                if groups[g].rows_left == 0 {
                                    events
                                        .entry(groups[g].local_ready)
                                        .or_default()
                                        .push(Event::FinishGroup(g));
                                }
                            } else {
                                groups[g].rmw_left -= 1;
                                if groups[g].rmw_left == 0 {
                                    let end = cp.reserve(
                                        now,
                                        control(f.completion_cycles, f),
                                        0,
                                        &groups[g].ids,
                                        &mut log,
                                    );
                                    stats.completion_services += 1;
                                    events
                                        .entry(end)
                                        .or_default()
                                        .push(Event::Commit(groups[g].ids.clone()));
                                }
                            }
                        }
                        Event::Commit(ids) => {
                            for id in ids {
                                let t = &mut tasks[id];
                                t.trace.commit_cycle = now;
                                let j = &r.jobs[t.key.job];
                                for &row in &t.trace.rows {
                                    let idx = t.key.nt * j.m + row;
                                    assert!(in_use[t.key.job][idx]);
                                    assert_eq!(progress[t.key.job][idx], t.key.kt);
                                    in_use[t.key.job][idx] = false;
                                    progress[t.key.job][idx] += 1;
                                    if (t.key.kt + 1) * r.k_lanes < j.k {
                                        ready.add(
                                            Key {
                                                kt: t.key.kt + 1,
                                                ..t.key
                                            },
                                            row,
                                        );
                                    } else {
                                        remaining[owner[t.key.job]] -= 1;
                                    }
                                }
                                pending[t.trace.core] -= 1;
                                active -= 1;
                            }
                        }
                        Event::FinishGroup(g) => {
                            let end = cp.reserve(
                                now,
                                control(f.completion_cycles, f),
                                0,
                                &groups[g].ids,
                                &mut log,
                            );
                            stats.completion_services += 1;
                            events.entry(end).or_default().push(Event::RetireGroup(g));
                        }
                        Event::RetireGroup(g) => {
                            assert_eq!(persistent.remove(&tasks[groups[g].ids[0]].key), Some(g));
                            open_cohorts -= 1;
                        }
                    }
                }
            }
            if phase == 1 {
                break;
            }
            let mut admitted = Vec::new();
            let mut provisional = BTreeSet::new();
            let mut preferred = None;
            for off in 0..nc {
                let c = (cursor + off) % nc;
                // Reserve result credit at admission: a bundled completion must
                // never wait for a member that cannot obtain result storage.
                if stages[c].len() >= f.stages_per_core
                    || active + open_cohorts + provisional.len() >= f.descriptors
                    || pending[c] + stages[c].len() >= cap
                {
                    continue;
                }
                let allowed = |key: &Key| {
                    (r.ownership == Ownership::TileStealing || owner[key.job] == c)
                        && private.as_ref().is_none_or(|p| {
                            ready.groups[key]
                                .iter()
                                .any(|&row| p.eligible(r, *key, row, c))
                        })
                };
                let mut candidates = Vec::new();
                let lookahead =
                    (f.ready_window * r.m_lanes[c] / r.m_lanes.iter().sum::<usize>()).max(1);
                let resume = ready
                    .fifo
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| allowed(k) && persistent.contains_key(k))
                    .take(lookahead);
                let fresh = ready
                    .fifo
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| allowed(k))
                    .take(lookahead);
                let mut seen = BTreeSet::new();
                for (idx, key) in resume.chain(fresh) {
                    if !seen.insert(idx) {
                        continue;
                    }
                    let existing = persistent.contains_key(key) || provisional.contains(key);
                    let new = f.control == Control::TileCohort && !existing;
                    if new && open_cohorts + provisional.len() >= f.descriptors / 2 {
                        continue;
                    }
                    if active + open_cohorts + provisional.len() + 1 + usize::from(new)
                        > f.descriptors
                    {
                        continue;
                    }
                    let hit = slots[c].iter().position(|s| s.key == Some(*key));
                    let slot = hit.or_else(|| {
                        slots[c]
                            .iter()
                            .enumerate()
                            .filter(|(_, s)| s.refs == 0 && s.ready <= now)
                            .min_by_key(|(i, s)| (s.key.is_some(), s.touched, *i))
                            .map(|(i, _)| i)
                    });
                    if let Some(slot) = slot {
                        let score = if f.selector == Selector::Affinity {
                            (
                                usize::from(hit.is_none()),
                                usize::from(preferred != Some(*key)),
                                idx,
                            )
                        } else {
                            (0, 0, idx)
                        };
                        candidates.push((
                            (
                                usize::from(f.control == Control::TileCohort && !existing),
                                score,
                            ),
                            idx,
                            slot,
                            hit.is_some(),
                        ));
                    }
                }
                let Some((_, idx, s, hit)) = candidates.into_iter().min_by_key(|x| x.0) else {
                    continue;
                };
                let candidate = ready.fifo[idx];
                stats.resume_selections += usize::from(persistent.contains_key(&candidate));
                let new_cohort = f.control == Control::TileCohort
                    && !persistent.contains_key(&candidate)
                    && !provisional.contains(&candidate);
                if active + open_cohorts + provisional.len() + 1 + usize::from(new_cohort)
                    > f.descriptors
                {
                    continue;
                }
                if new_cohort {
                    provisional.insert(candidate);
                }
                let (key, rows) = if let Some(p) = &mut private {
                    let key = ready.fifo[idx];
                    let rows: Vec<_> = ready.groups[&key]
                        .iter()
                        .copied()
                        .filter(|&row| p.eligible(r, key, row, c))
                        .take(r.m_lanes[c])
                        .collect();
                    // New rows may each fit in isolation but not together. Reserve
                    // each bounded accumulator entry before removing it from Ready.
                    let mut selected = Vec::new();
                    for row in rows {
                        if p.eligible(r, key, row, c) {
                            p.claim(r, key, row, c);
                            selected.push(row);
                        }
                    }
                    assert!(!selected.is_empty());
                    ready.fifo.remove(idx);
                    let available = ready.groups.get_mut(&key).unwrap();
                    for row in &selected {
                        assert!(available.remove(row));
                    }
                    if available.is_empty() {
                        ready.groups.remove(&key);
                    } else {
                        ready.fifo.push_back(key);
                    }
                    (key, selected)
                } else {
                    ready.take(idx, r.m_lanes[c])
                };
                // Keep a partially consumed cohort visible to the finite
                // lookahead. Rotating it to the end would silently defeat
                // both multicast and temporal weight reuse on long N queues.
                if f.selector == Selector::Affinity && ready.groups.contains_key(&key) {
                    let pos = ready.fifo.iter().position(|&k| k == key).unwrap();
                    ready.fifo.remove(pos);
                    ready.fifo.push_front(key);
                }
                preferred = Some(key);
                let j = &r.jobs[key.job];
                for &row in &rows {
                    let idx = key.nt * j.m + row;
                    assert!(!in_use[key.job][idx]);
                    assert_eq!(progress[key.job][idx], key.kt);
                    in_use[key.job][idx] = true;
                }
                let slot = &mut slots[c][s];
                if hit {
                    stats.cache_hits += 1;
                } else {
                    stats.cache_misses += 1;
                    slot.key = Some(key);
                    slot.ready = u64::MAX;
                    slot.values.clear();
                }
                slot.refs += 1;
                slot.touched = now;
                let nv = r.n_lanes.min(j.n - key.nt * r.n_lanes);
                let kv = r.k_lanes.min(j.k - key.kt * r.k_lanes);
                let id = tasks.len();
                tasks.push(Task {
                    key,
                    miss: !hit,
                    values: Vec::new(),
                    activation: None,
                    trace: Trace {
                        request: id,
                        group: usize::MAX,
                        core: c,
                        expert: j.expert,
                        rows: rows.clone(),
                        n_start: key.nt * r.n_lanes,
                        k_start: key.kt * r.k_lanes,
                        valid_n: nv,
                        valid_k: kv,
                        admitted: now,
                        descriptor_ready: u64::MAX,
                        weight_ready: u64::MAX,
                        activation_ready: u64::MAX,
                        issue_cycle: u64::MAX,
                        mac_done: u64::MAX,
                        rmw_done: u64::MAX,
                        commit_cycle: u64::MAX,
                        useful_macs: (rows.len() * nv * kv) as u64,
                        issued_mac_slots: (r.m_lanes[c] * r.n_lanes * r.k_lanes) as u64,
                        slot: s,
                        cache_hit: hit,
                        operand_ready: None,
                    },
                });
                stages[c].push_back(id);
                admitted.push(id);
                active += 1;
                cores[c].stage_peak = cores[c].stage_peak.max(stages[c].len());
            }
            let mut bundles: Vec<Vec<usize>> = Vec::new();
            let any_admitted = !admitted.is_empty();
            for id in admitted {
                let existing = if f.control != Control::Invocation {
                    bundles
                        .iter()
                        .position(|b| tasks[b[0]].key == tasks[id].key)
                } else {
                    None
                };
                if let Some(g) = existing {
                    bundles[g].push(id);
                } else {
                    bundles.push(vec![id]);
                }
            }
            for ids in bundles {
                let key = tasks[ids[0]].key;
                if f.control == Control::TileCohort
                    && let Some(&g) = persistent.get(&key)
                {
                    for &id in &ids {
                        tasks[id].trace.group = g;
                    }
                    groups[g].ids.extend(&ids);
                    events
                        .entry(now.max(groups[g].header_ready))
                        .or_default()
                        .push(Event::Dispatch(ids));
                    continue;
                }
                let g = groups.len();
                for &id in &ids {
                    tasks[id].trace.group = g;
                }
                let end = cp.reserve(now, control(f.issue_cycles, f), 0, &ids, &mut log);
                stats.issue_services += 1;
                groups.push(Group {
                    rmw_left: ids.len(),
                    ids: ids.clone(),
                    rows_left: r.jobs[key.job].m,
                    header_ready: end,
                    local_ready: 0,
                });
                if f.control == Control::TileCohort {
                    persistent.insert(key, g);
                    open_cohorts += 1;
                }
                events.entry(end).or_default().push(Event::Dispatch(ids));
            }
            if any_admitted {
                cursor = (cursor + 1) % nc;
            }
            stats.descriptor_peak = stats.descriptor_peak.max(active);
            stats.cohort_descriptor_peak = stats.cohort_descriptor_peak.max(open_cohorts);
            stats.total_metadata_peak = stats.total_metadata_peak.max(active + open_cohorts);
            let wb = slots.iter().flatten().filter(|s| s.key.is_some()).count() * weight_size;
            let xb = stages
                .iter()
                .enumerate()
                .map(|(c, s)| s.len() * r.m_lanes[c] * r.k_lanes * 2)
                .sum::<usize>();
            let rb = pending
                .iter()
                .enumerate()
                .map(|(c, n)| (n + stages[c].len()) * r.m_lanes[c] * r.n_lanes * 4)
                .sum::<usize>();
            if let Some(p) = &mut private {
                for c in 0..nc {
                    p.report.weight_peak_bytes[c] = p.report.weight_peak_bytes[c]
                        .max(slots[c].iter().filter(|s| s.key.is_some()).count() * weight_size);
                    p.report.activation_peak_bytes[c] = p.report.activation_peak_bytes[c]
                        .max(stages[c].len() * r.m_lanes[c] * r.k_lanes * 2);
                    p.report.result_peak_bytes[c] = p.report.result_peak_bytes[c]
                        .max((pending[c] + stages[c].len()) * r.m_lanes[c] * r.n_lanes * 4);
                }
            }
            stats.weight_storage_peak_bytes = stats.weight_storage_peak_bytes.max(wb);
            stats.activation_storage_peak_bytes = stats.activation_storage_peak_bytes.max(xb);
            stats.result_storage_peak_bytes = stats.result_storage_peak_bytes.max(rb);
            stats.simultaneous_operand_result_peak_bytes = stats
                .simultaneous_operand_result_peak_bytes
                .max(wb + xb + rb);
        }
        if remaining.iter().all(|&x| x == 0) && active == 0 && open_cohorts == 0 {
            break;
        }
        if active == 0 && events.is_empty() && !ready.fifo.is_empty() {
            return Err(
                "no admissible output: private accumulator capacity or ownership deadlock".into(),
            );
        }
        // Local SRAM reads are real services, after both operand writes finish.
        if let Some(p) = &mut private {
            for c in 0..nc {
                if let Some(&id) = stages[c].iter().find(|&&id| {
                    let t = &tasks[id];
                    t.trace.operand_ready.is_none()
                        && t.trace.descriptor_ready <= now
                        && t.trace.activation_ready <= now
                        && slots[c][t.trace.slot].ready <= now
                }) {
                    let t = &mut tasks[id];
                    let w = p.weight_read[c].reserve_data(
                        now,
                        aligned(t.trace.valid_n * t.trace.valid_k * 2),
                        (p.config.weight_read_bpc[c], f.zero_weight_time, false),
                        &[id],
                        &mut log,
                    );
                    let x = p.activation_read[c].reserve_data(
                        now,
                        aligned(t.trace.rows.len() * t.trace.valid_k * 2),
                        (
                            p.config.activation_read_bpc[c],
                            f.zero_activation_time,
                            false,
                        ),
                        &[id],
                        &mut log,
                    );
                    t.trace.operand_ready = Some(w.max(x));
                }
            }
        }
        let mut issued = vec![false; nc];
        for c in 0..nc {
            if now < next_issue[c] || pending[c] >= cap {
                continue;
            }
            let pos = stages[c].iter().position(|&id| {
                let t = &tasks[id];
                t.trace.descriptor_ready <= now
                    && t.trace.activation_ready <= now
                    && slots[c][t.trace.slot].ready <= now
                    && (private.is_none() || t.trace.operand_ready.is_some_and(|end| end <= now))
            });
            let Some(pos) = pos else {
                continue;
            };
            let id = stages[c].remove(pos).unwrap();
            let t = &mut tasks[id];
            let slot = &mut slots[c][t.trace.slot];
            assert_eq!(slot.key, Some(t.key));
            assert!(slot.refs > 0);
            t.trace.weight_ready = slot.ready;
            t.trace.issue_cycle = now;
            t.trace.mac_done = now + r.result_latency_cycles;
            if r.verify_values {
                if private.is_some() {
                    assert!(t.activation.is_some());
                }
                t.values = multiply(r, t, &slot.values, external);
                t.activation = None;
            }
            slot.refs -= 1;
            slot.touched = now;
            if !f.retain_weights && slot.refs == 0 {
                slot.key = None;
                slot.values.clear();
            }
            pending[c] += 1;
            cores[c].pipeline_peak = cores[c].pipeline_peak.max(pending[c]);
            cores[c].invocations += 1;
            cores[c].useful_macs += t.trace.useful_macs;
            cores[c].issued_mac_slots += t.trace.issued_mac_slots;
            next_issue[c] = now + r.issue_interval_cycles;
            issued[c] = true;
            events
                .entry(t.trace.mac_done)
                .or_default()
                .push(Event::MacDone(id));
        }
        let rb = pending
            .iter()
            .enumerate()
            .map(|(c, n)| (n + stages[c].len()) * r.m_lanes[c] * r.n_lanes * 4)
            .sum::<usize>();
        stats.result_storage_peak_bytes = stats.result_storage_peak_bytes.max(rb);
        for c in 0..nc {
            let state = if issued[c] || now < next_issue[c] {
                "issue_interval"
            } else if pending[c] >= cap {
                "result_backpressure"
            } else if let Some(&id) = stages[c].front() {
                let t = &tasks[id];
                if t.trace.descriptor_ready > now {
                    "control_issue"
                } else if slots[c][t.trace.slot].ready > now {
                    "weight_delivery_or_install"
                } else if t.trace.activation_ready > now {
                    "activation"
                } else if private.is_some() && t.trace.operand_ready.is_none_or(|end| end > now) {
                    "local_operand_read"
                } else {
                    "ready_behind_other_stage"
                }
            } else if active >= f.descriptors {
                "descriptor_credit"
            } else if slots[c].iter().all(|s| s.refs > 0 || s.ready > now) {
                "weight_slot"
            } else if (r.ownership == Ownership::PinnedExpert && remaining[c] == 0)
                || remaining.iter().sum::<usize>() == 0
            {
                "finished_idle"
            } else {
                "dependency_or_ready_window"
            };
            *cores[c].states.entry(state.into()).or_default() += 1;
        }
        now += 1;
        if now > 20_000_000 {
            return Err("fabric cycle safety limit exceeded".into());
        }
    }
    if let Some(src) = source.as_deref()
        && !src.drained()
    {
        return Err("native weight source not drained".into());
    }
    assert!(events.is_empty() && ready.fifo.is_empty() && stages.iter().all(|s| s.is_empty()));
    assert!(pending.iter().all(|&n| n == 0) && slots.iter().flatten().all(|s| s.refs == 0));
    assert!(in_use.iter().flatten().all(|&b| !b));
    assert!(
        stats.weight_storage_peak_bytes <= stats.weight_budget_bytes
            && stats.activation_storage_peak_bytes <= stats.activation_budget_bytes
            && stats.result_storage_peak_bytes <= stats.result_budget_bytes
            && stats.total_metadata_peak <= f.descriptors
    );
    for c in &cores {
        assert_eq!(c.states.values().sum::<u64>(), now);
    }
    if let Some(p) = &mut private {
        values = p.gather(r);
        p.audit(r, f, &tasks, &log);
    }
    if let Some(ref out) = values {
        for (index, (j, v)) in r.jobs.iter().zip(out).enumerate() {
            if !v
                .iter()
                .zip(external.map_or_else(|| reference(j), |d| d.reference(r, index)))
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err("fabric numerical reference mismatch".into());
            }
        }
    }
    let useful = r.jobs.iter().map(|j| (j.m * j.n * j.k) as u64).sum::<u64>();
    assert_eq!(useful, cores.iter().map(|c| c.useful_macs).sum::<u64>());
    let mut trace_hash = Sha256::new();
    let mut port_hash = Sha256::new();
    let mut last_issue = vec![None; nc];
    // Audit in issue order, since admissions can bypass a waiting stage.
    let mut ordered: Vec<_> = tasks.iter().collect();
    ordered.sort_by_key(|t| (t.trace.issue_cycle, t.trace.core));
    for t in ordered {
        let x = &t.trace;
        assert!(x.admitted <= x.descriptor_ready && x.descriptor_ready <= x.activation_ready);
        assert!(
            x.issue_cycle >= x.descriptor_ready
                && x.issue_cycle >= x.activation_ready
                && x.issue_cycle >= x.weight_ready
                && x.operand_ready.is_none_or(|end| x.issue_cycle >= end)
        );
        assert_eq!(x.mac_done, x.issue_cycle + r.result_latency_cycles);
        assert!(x.mac_done <= x.rmw_done && x.rmw_done <= x.commit_cycle && x.commit_cycle <= now);
        if let Some(prev) = last_issue[x.core] {
            assert!(x.issue_cycle >= prev + r.issue_interval_cycles);
        }
        last_issue[x.core] = Some(x.issue_cycle);
    }
    for t in &tasks {
        trace_hash.update(serde_json::to_vec(&t.trace).unwrap());
    }
    let mut last_port: BTreeMap<(&str, usize), u64> = BTreeMap::new();
    let mut last_byte: BTreeMap<(&str, usize), u64> = BTreeMap::new();
    let mut port_bytes: BTreeMap<&str, u64> = BTreeMap::new();
    for s in &log {
        assert!(s.start >= s.released && s.end >= s.start && s.end <= now);
        let prev = last_port.entry((&s.resource, s.port)).or_default();
        let busy_start = s.start.max(*prev);
        if let Some(begin) = s.byte_start {
            let finish = s.byte_end.unwrap();
            let rate = s.byte_rate.unwrap();
            let previous_byte = last_byte.entry((&s.resource, s.port)).or_default();
            assert!(begin >= *previous_byte);
            *previous_byte = finish;
            assert_eq!(finish - begin, s.bytes);
            assert_eq!(s.start, begin / rate);
            assert_eq!(s.end, finish.div_ceil(rate));
        } else {
            assert!(s.start >= *prev);
        }
        *prev = s.end;
        *port_bytes.entry(&s.resource).or_default() += s.bytes;
        let cycles = s.end - busy_start;
        match s.resource.as_str() {
            "control" => stats.control_service_cycles += cycles,
            "weight" => stats.weight_service_cycles += cycles,
            "activation" => stats.activation_service_cycles += cycles,
            "accumulator" => stats.accumulator_service_cycles += cycles,
            "local_weight_read"
            | "local_weight_write"
            | "local_activation_read"
            | "local_activation_write" => {}
            _ => unreachable!(),
        }
        port_hash.update(serde_json::to_vec(s).unwrap());
    }
    assert_eq!(
        port_bytes.get("weight").copied().unwrap_or(0),
        stats.source_weight_bytes
    );
    assert_eq!(
        port_bytes.get("activation").copied().unwrap_or(0),
        stats.activation_bytes
    );
    assert_eq!(
        port_bytes.get("accumulator").copied().unwrap_or(0),
        stats.accumulator_rmw_bytes
    );
    assert_eq!(
        stats.control_service_cycles,
        if f.zero_control_time {
            0
        } else {
            stats.issue_services * f.issue_cycles
                + stats.install_services * f.install_cycles
                + stats.completion_services * f.completion_cycles
        }
    );
    let invocation_audit_count = tasks.len();
    let service_audit_count = log.len();
    Ok(Output {
        name: r.name.clone(),
        scope: if source.is_some() {
            "native HBM2 weight timing with finite spatial-M fabric; BF16; activation on-chip; GEMM only; not layer/model E2E".into()
        } else {
            "finite decoded operand-interface model; assumed timing; no native HBM/PPA/full-model claim".into()
        },
        total_cycles: now,
        total_multipliers: r.total_multiplier_budget,
        useful_macs: useful,
        issued_mac_slots: cores.iter().map(|c| c.issued_mac_slots).sum(),
        numerical_bit_exact: r.verify_values.then_some(true),
        output_fp32_bits: values.as_ref().map(|o| {
            o.iter()
                .map(|v| v.iter().map(|x| x.to_bits()).collect())
                .collect()
        }),
        output_bf16_bits: values.as_ref().map(|o| {
            o.iter()
                .map(|v| v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect())
                .collect()
        }),
        drained: true,
        private_memories: private.map(|p| p.report),
        stats,
        cores,
        trace: if r.record_trace {
            tasks.into_iter().map(|t| t.trace).collect()
        } else {
            Vec::new()
        },
        services: if r.record_trace { log } else { Vec::new() },
        invocation_audit_count,
        service_audit_count,
        invocation_sha256: format!("{:x}", trace_hash.finalize()),
        service_sha256: format!("{:x}", port_hash.finalize()),
    })
}
#[cfg(test)]
mod tests;
