//! Finite pipelined spatial-M datapath, with actual numerical GEMM execution.
//! Hardware latency is an explicit model input, not inferred from wave count.
use half::bf16;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[allow(dead_code)]
pub mod fabric;
pub mod operands;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    PinnedExpert,
    TileStealing,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub expert: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub seed: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub name: String,
    pub m_lanes: Vec<usize>,
    pub n_lanes: usize,
    pub k_lanes: usize,
    pub total_multiplier_budget: usize,
    pub result_latency_cycles: u64,
    pub issue_interval_cycles: u64,
    pub ownership: Ownership,
    pub verify_values: bool,
    pub record_trace: bool,
    pub jobs: Vec<Job>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct CoreReport {
    pub m_lanes: usize,
    pub multipliers: usize,
    pub assigned_experts: Vec<usize>,
    pub invocations: u64,
    pub useful_macs: u64,
    pub issued_mac_slots: u64,
    pub tail_mac_slots: u64,
    pub pipeline_capacity: usize,
    pub pipeline_peak: usize,
    pub pipeline_result_register_bytes: usize,
    pub pipeline_metadata_entries: usize,
    /// No eligible invocation: includes K feedback and final result drain.
    pub no_ready_invocation_cycles: u64,
    pub no_remaining_work_cycles: u64,
    pub issue_interval_wait_cycles: u64,
    pub pipeline_full_cycles: u64,
    pub issue_cycles: Vec<u64>,
    pub last_completion_cycle: u64,
    pub logical_activation_elements: u64,
    pub logical_weight_elements: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Invocation {
    pub core: usize,
    pub expert: usize,
    pub rows: Vec<usize>,
    pub n_start: usize,
    pub k_start: usize,
    pub valid_n: usize,
    pub valid_k: usize,
    pub issue_cycle: u64,
    pub completion_cycle: u64,
    pub useful_macs: u64,
    pub issued_mac_slots: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Report {
    pub schema_version: u32,
    pub name: String,
    pub evidence: String,
    pub hardware_model: String,
    pub total_cycles: u64,
    pub total_multipliers: usize,
    pub useful_macs: u64,
    pub issued_mac_slots: u64,
    pub tail_mac_slots: u64,
    pub invocation_utilization: f64,
    pub physical_mac_slot_utilization: f64,
    pub effective_issue_capacity_utilization: f64,
    pub total_invocations: u64,
    pub distinct_issue_times: usize,
    pub global_timing_advance_steps: u64,
    pub dependency_order_checks: u64,
    pub requests_drained: bool,
    pub numerical_bit_exact: Option<bool>,
    pub output_fp32_bits: Option<Vec<Vec<u32>>>,
    pub output_bf16_bits: Option<Vec<Vec<u16>>>,
    pub cores: Vec<CoreReport>,
    pub trace: Vec<Invocation>,
}

// One output row within one N tile is the unit of K dependency ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    job: usize,
    nt: usize,
    kt: usize,
}

#[derive(Default)]
struct Ready {
    groups: BTreeMap<Key, BTreeSet<usize>>,
    fifo: VecDeque<Key>,
}

impl Ready {
    fn add(&mut self, key: Key, row: usize) {
        if !self.groups.contains_key(&key) {
            self.fifo.push_back(key);
        }
        assert!(self.groups.entry(key).or_default().insert(row));
    }

    fn candidate(&self, core: usize, owners: &[usize], mode: Ownership) -> Option<usize> {
        self.fifo
            .iter()
            .position(|k| mode == Ownership::TileStealing || owners[k.job] == core)
    }

    fn take(&mut self, index: usize, width: usize) -> (Key, Vec<usize>) {
        let key = self.fifo.remove(index).unwrap();
        let available = self.groups.get_mut(&key).unwrap();
        let mut rows = Vec::new();
        for _ in 0..width {
            if let Some(row) = available.pop_first() {
                rows.push(row);
            }
        }
        assert!(!rows.is_empty());
        if available.is_empty() {
            self.groups.remove(&key);
        } else {
            self.fifo.push_back(key);
        }
        (key, rows)
    }
}

struct Pending {
    core: usize,
    key: Key,
    rows: Vec<usize>,
    // A bounded hardware result tile, not an unbounded future accumulator.
    values: Option<Vec<f32>>,
}

fn validate(r: &Request) -> Result<(), String> {
    if r.m_lanes.is_empty()
        || r.m_lanes.len() > 32
        || r.m_lanes.iter().any(|&m| m == 0 || m > 256)
        || r.n_lanes == 0
        || r.n_lanes > 512
        || !r.k_lanes.is_power_of_two()
        || r.k_lanes > 4096
        || r.jobs.is_empty()
        || r.issue_interval_cycles == 0
        || r.result_latency_cycles < r.issue_interval_cycles
        || r.result_latency_cycles > 4096
    {
        return Err("invalid finite geometry or pipeline timing".into());
    }
    let multipliers = r.m_lanes.iter().sum::<usize>() * r.n_lanes * r.k_lanes;
    if multipliers != r.total_multiplier_budget {
        return Err("physical M*N*K multiplier budget mismatch".into());
    }
    let mut ids = BTreeSet::new();
    let mut macs = 0u128;
    let mut outputs = 0u128;
    for j in &r.jobs {
        if j.m == 0
            || j.n == 0
            || j.k == 0
            || j.m > 65536
            || j.n > 65536
            || j.k > 65536
            || !ids.insert(j.expert)
        {
            return Err("invalid shape or repeated expert ID".into());
        }
        macs += j.m as u128 * j.n as u128 * j.k as u128;
        outputs += j.m as u128 * j.n as u128;
    }
    if macs > 1_000_000_000_000 || outputs > 16_000_000 {
        return Err("experiment safety bound exceeded".into());
    }
    Ok(())
}

fn owners(r: &Request) -> Vec<usize> {
    let mut order: Vec<_> = (0..r.jobs.len()).collect();
    order.sort_by_key(|&i| {
        (
            std::cmp::Reverse(r.jobs[i].m * r.jobs[i].n * r.jobs[i].k),
            i,
        )
    });
    let mut load = vec![0u64; r.m_lanes.len()];
    let mut owner = vec![0; r.jobs.len()];
    for i in order {
        let j = &r.jobs[i];
        let (core, next) = r
            .m_lanes
            .iter()
            .enumerate()
            .map(|(c, &m)| {
                let waves = j.m.div_ceil(m) * j.n.div_ceil(r.n_lanes) * j.k.div_ceil(r.k_lanes);
                (c, load[c] + waves as u64)
            })
            .min_by_key(|&(c, v)| (v, c))
            .unwrap();
        owner[i] = core;
        load[core] = next;
    }
    owner
}

fn x_value(j: &Job, row: usize, k: usize) -> f32 {
    let v = ((row as u64 * 7 + k as u64 * 3 + j.seed as u64) % 17) as i32 - 8;
    bf16::from_f32(v as f32 / 8.0).to_f32()
}

fn w_value(j: &Job, k: usize, col: usize) -> f32 {
    let v = ((k as u64 * 5 + col as u64 * 11 + j.seed as u64 * 7 + j.expert as u64 * 3) % 19)
        as i32
        - 9;
    bf16::from_f32(v as f32 / 16.0).to_f32()
}

fn tree_reduce(mut values: Vec<f32>) -> f32 {
    assert!(values.len().is_power_of_two());
    let mut count = values.len();
    while count > 1 {
        for i in 0..count / 2 {
            values[i] = values[2 * i] + values[2 * i + 1];
        }
        count /= 2;
    }
    values[0]
}

fn partial(r: &Request, key: Key, rows: &[usize]) -> Vec<f32> {
    let j = &r.jobs[key.job];
    let n0 = key.nt * r.n_lanes;
    let k0 = key.kt * r.k_lanes;
    let mut out = Vec::new();
    for &row in rows {
        for col in n0..(n0 + r.n_lanes).min(j.n) {
            let mut products = vec![0.0; r.k_lanes];
            for (kk, product) in products
                .iter_mut()
                .enumerate()
                .take((j.k - k0).min(r.k_lanes))
            {
                *product = x_value(j, row, k0 + kk) * w_value(j, k0 + kk, col);
            }
            out.push(tree_reduce(products));
        }
    }
    out
}

fn reference(j: &Job) -> Vec<f32> {
    let mut out = vec![0.0; j.m * j.n];
    for row in 0..j.m {
        for col in 0..j.n {
            let mut sum = 0.0;
            for k in 0..j.k {
                sum += x_value(j, row, k) * w_value(j, k, col);
            }
            out[row * j.n + col] = sum;
        }
    }
    out
}

pub fn run(r: &Request) -> Result<Report, String> {
    validate(r)?;
    let owner = owners(r);
    let capacity = r.result_latency_cycles.div_ceil(r.issue_interval_cycles) as usize;
    let mut reports: Vec<_> = r
        .m_lanes
        .iter()
        .enumerate()
        .map(|(c, &m)| CoreReport {
            m_lanes: m,
            multipliers: m * r.n_lanes * r.k_lanes,
            assigned_experts: if r.ownership == Ownership::PinnedExpert {
                r.jobs
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| owner[i] == c)
                    .map(|(_, j)| j.expert)
                    .collect()
            } else {
                Vec::new()
            },
            pipeline_capacity: capacity,
            pipeline_result_register_bytes: capacity * m * r.n_lanes * 4,
            pipeline_metadata_entries: capacity,
            ..Default::default()
        })
        .collect();
    let mut ready = Ready::default();
    let mut progress = Vec::new();
    let mut inflight_rows = Vec::new();
    let mut unfinished = vec![0usize; r.m_lanes.len()];
    for (i, j) in r.jobs.iter().enumerate() {
        let count = j.m * j.n.div_ceil(r.n_lanes);
        progress.push(vec![0usize; count]);
        inflight_rows.push(vec![false; count]);
        unfinished[owner[i]] += count;
        for nt in 0..j.n.div_ceil(r.n_lanes) {
            for row in 0..j.m {
                ready.add(Key { job: i, nt, kt: 0 }, row);
            }
        }
    }
    let mut outputs = r.verify_values.then(|| {
        r.jobs
            .iter()
            .map(|j| vec![0.0f32; j.m * j.n])
            .collect::<Vec<_>>()
    });
    let mut events: BTreeMap<u64, Vec<Pending>> = BTreeMap::new();
    let mut next_issue = vec![0u64; r.m_lanes.len()];
    let mut pending = vec![0usize; r.m_lanes.len()];
    let mut trace = Vec::new();
    let mut distinct_times = BTreeSet::new();
    let mut now = 0u64;
    let mut checks = 0u64;
    let mut advances = 0u64;
    loop {
        if let Some(completions) = events.remove(&now) {
            for done in completions {
                let key = done.key;
                let j = &r.jobs[key.job];
                let n0 = key.nt * r.n_lanes;
                let nv = (j.n - n0).min(r.n_lanes);
                for (ri, &row) in done.rows.iter().enumerate() {
                    let output = key.nt * j.m + row;
                    assert!(inflight_rows[key.job][output]);
                    assert_eq!(progress[key.job][output], key.kt);
                    checks += 1;
                    if let Some(ref mut values) = outputs {
                        for nc in 0..nv {
                            values[key.job][row * j.n + n0 + nc] +=
                                done.values.as_ref().unwrap()[ri * nv + nc];
                        }
                    }
                    inflight_rows[key.job][output] = false;
                    progress[key.job][output] += 1;
                    if (key.kt + 1) * r.k_lanes < j.k {
                        ready.add(
                            Key {
                                kt: key.kt + 1,
                                ..key
                            },
                            row,
                        );
                    } else {
                        unfinished[owner[key.job]] -= 1;
                    }
                }
                pending[done.core] -= 1;
                reports[done.core].last_completion_cycle = now;
            }
        }
        for core in 0..r.m_lanes.len() {
            if now < next_issue[core] || pending[core] == capacity {
                continue;
            }
            let Some(index) = ready.candidate(core, &owner, r.ownership) else {
                continue;
            };
            let (key, rows) = ready.take(index, r.m_lanes[core]);
            let j = &r.jobs[key.job];
            for &row in &rows {
                let output = key.nt * j.m + row;
                assert!(!inflight_rows[key.job][output]);
                assert_eq!(progress[key.job][output], key.kt);
                inflight_rows[key.job][output] = true;
            }
            let nv = (j.n - key.nt * r.n_lanes).min(r.n_lanes);
            let kv = (j.k - key.kt * r.k_lanes).min(r.k_lanes);
            let useful = (rows.len() * nv * kv) as u64;
            let padded = reports[core].multipliers as u64;
            let completion = now + r.result_latency_cycles;
            if r.record_trace {
                trace.push(Invocation {
                    core,
                    expert: j.expert,
                    rows: rows.clone(),
                    n_start: key.nt * r.n_lanes,
                    k_start: key.kt * r.k_lanes,
                    valid_n: nv,
                    valid_k: kv,
                    issue_cycle: now,
                    completion_cycle: completion,
                    useful_macs: useful,
                    issued_mac_slots: padded,
                });
                reports[core].issue_cycles.push(now);
            }
            let values = r.verify_values.then(|| partial(r, key, &rows));
            events.entry(completion).or_default().push(Pending {
                core,
                key,
                rows,
                values,
            });
            pending[core] += 1;
            next_issue[core] = now + r.issue_interval_cycles;
            let report = &mut reports[core];
            report.pipeline_peak = report.pipeline_peak.max(pending[core]);
            report.invocations += 1;
            report.useful_macs += useful;
            report.issued_mac_slots += padded;
            report.tail_mac_slots += padded - useful;
            report.logical_activation_elements += useful / nv as u64;
            report.logical_weight_elements += (nv * kv) as u64;
            distinct_times.insert(now);
        }
        if ready.fifo.is_empty() && events.is_empty() {
            break;
        }
        let mut next = events
            .first_key_value()
            .map(|(&t, _)| t)
            .unwrap_or(u64::MAX);
        for core in 0..r.m_lanes.len() {
            if pending[core] < capacity && ready.candidate(core, &owner, r.ownership).is_some() {
                assert!(next_issue[core] > now);
                next = next.min(next_issue[core]);
            }
        }
        if next == u64::MAX || next <= now {
            return Err("spatial pipeline deadlock".into());
        }
        let delta = next - now;
        for core in 0..r.m_lanes.len() {
            let report = &mut reports[core];
            // Exclusive issue-side state partition, not an additive memory/MAC profile.
            let occupied = next_issue[core].saturating_sub(now).min(delta);
            report.issue_interval_wait_cycles += occupied;
            let remainder = delta - occupied;
            if pending[core] == capacity {
                report.pipeline_full_cycles += remainder;
            } else if (r.ownership == Ownership::PinnedExpert && unfinished[core] == 0)
                || unfinished.iter().sum::<usize>() == 0
            {
                report.no_remaining_work_cycles += remainder;
            } else {
                report.no_ready_invocation_cycles += remainder;
            }
        }
        now = next;
        advances += 1;
    }
    assert!(pending.iter().all(|&x| x == 0) && unfinished.iter().all(|&x| x == 0));
    assert!(inflight_rows.iter().flatten().all(|&x| !x));
    for core in &reports {
        assert_eq!(
            core.issue_interval_wait_cycles
                + core.pipeline_full_cycles
                + core.no_remaining_work_cycles
                + core.no_ready_invocation_cycles,
            now
        );
    }
    let useful: u64 = r.jobs.iter().map(|j| (j.m * j.n * j.k) as u64).sum();
    assert_eq!(reports.iter().map(|c| c.useful_macs).sum::<u64>(), useful);
    let issued: u64 = reports.iter().map(|c| c.issued_mac_slots).sum();
    let output_fp32_bits = outputs.as_ref().map(|out| {
        out.iter()
            .map(|v| v.iter().map(|x| x.to_bits()).collect())
            .collect()
    });
    let output_bf16_bits = outputs.as_ref().map(|out| {
        out.iter()
            .map(|v| v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect())
            .collect()
    });
    if let Some(ref out) = outputs {
        for (j, values) in r.jobs.iter().zip(out) {
            let reference = reference(j);
            if !values
                .iter()
                .zip(reference)
                .all(|(x, y)| x.to_bits() == y.to_bits())
            {
                return Err(format!(
                    "independent scalar reference differs for expert {}",
                    j.expert
                ));
            }
        }
    }
    Ok(Report {
        schema_version: 1, name: r.name.clone(),
        evidence: if r.verify_values { "compute_only_numerical_synthetic_bf16" } else { "compute_only_shape_timing_no_numerical_execution" }.into(),
        hardware_model: "physical m*n*k multipliers; ideal operand/accumulator interfaces; bounded result pipeline; one expert per invocation; explicit L/II; no SRAM/HBM/PPA claim".into(),
        total_cycles: now, total_multipliers: r.total_multiplier_budget, useful_macs: useful,
        issued_mac_slots: issued, tail_mac_slots: issued - useful,
        invocation_utilization: useful as f64 / issued as f64,
        physical_mac_slot_utilization: useful as f64 / (r.total_multiplier_budget as f64 * now as f64),
        effective_issue_capacity_utilization: useful as f64 / (r.total_multiplier_budget as f64 * now.div_ceil(r.issue_interval_cycles) as f64),
        total_invocations: reports.iter().map(|c| c.invocations).sum(),
        distinct_issue_times: distinct_times.len(), global_timing_advance_steps: advances,
        dependency_order_checks: checks, requests_drained: true,
        numerical_bit_exact: r.verify_values.then_some(true), output_fp32_bits, output_bf16_bits,
        cores: reports, trace,
    })
}

#[cfg(test)]
mod tests;
