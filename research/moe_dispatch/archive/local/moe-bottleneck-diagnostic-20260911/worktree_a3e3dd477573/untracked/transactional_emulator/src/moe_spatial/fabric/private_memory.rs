//! Private SRAMs and persistent output ownership. Physical storage is charged
//! when reserved, not when a host Vec happens to allocate. Host FP32 copies of
//! BF16 operands are numerical representations, not extra simulated SRAM.
use super::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateConfig {
    pub weight_slots: Vec<usize>,
    pub accumulator_bytes: Vec<usize>,
    pub accumulator_bpc: Vec<u64>,
    pub weight_read_bpc: Vec<u64>,
    pub weight_write_bpc: Vec<u64>,
    pub activation_read_bpc: Vec<u64>,
    pub activation_write_bpc: Vec<u64>,
    pub total_weight_read_bpc: u64,
    pub total_weight_write_bpc: u64,
}
#[derive(Clone, Debug, Serialize, PartialEq, Default)]
pub struct PrivateReport {
    pub weight_budget_bytes: Vec<usize>,
    pub activation_budget_bytes: Vec<usize>,
    pub result_budget_bytes: Vec<usize>,
    pub accumulator_budget_bytes: Vec<usize>,
    pub weight_peak_bytes: Vec<usize>,
    pub activation_peak_bytes: Vec<usize>,
    pub result_peak_bytes: Vec<usize>,
    pub accumulator_peak_bytes: Vec<usize>,
    pub accumulator_metadata_bytes: Vec<usize>,
    pub owned_output_tiles: Vec<usize>,
    pub cross_core_partial_sum_transfers: u64,
    pub ownership_verified: bool,
    pub owner_sha256: String,
    pub local_service_cycles: BTreeMap<String, Vec<u64>>,
    pub local_service_bytes: BTreeMap<String, Vec<u64>>,
}
type OutputKey = (usize, usize, usize); // job, N tile, M row
pub(super) struct PrivateState {
    pub config: PrivateConfig,
    pub report: PrivateReport,
    pub weight_read: Vec<Port>,
    pub weight_write: Vec<Port>,
    pub activation_read: Vec<Port>,
    pub activation_write: Vec<Port>,
    pub accumulator: Vec<Port>,
    pub(super) owners: BTreeMap<OutputKey, usize>,
    values: Vec<BTreeMap<OutputKey, Vec<f32>>>,
}
fn ports(name: &'static str, nc: usize) -> Vec<Port> {
    (0..nc)
        .map(|c| {
            let mut p = Port::new(name, 1);
            p.port_base = c;
            p
        })
        .collect()
}
impl PrivateState {
    pub fn new(
        config: PrivateConfig,
        r: &Request,
        f: &Fabric,
        pinned: &[usize],
    ) -> Result<Self, String> {
        let nc = r.m_lanes.len();
        let vectors = [
            &config.accumulator_bpc,
            &config.weight_read_bpc,
            &config.weight_write_bpc,
            &config.activation_read_bpc,
            &config.activation_write_bpc,
        ];
        if vectors
            .iter()
            .any(|v| v.len() != nc || v.iter().any(|&x| x == 0 || x > (1 << 30)))
            || config.weight_slots.len() != nc
            || config.weight_slots.iter().any(|&s| s == 0 || s > 4096)
            || config.accumulator_bytes.len() != nc
            || config.accumulator_bytes.contains(&0)
            || config.weight_slots.iter().sum::<usize>()
                != r.m_lanes.iter().sum::<usize>() * f.slots_per_m
            || config.accumulator_bytes.iter().sum::<usize>() != f.accumulator_bytes
            || config.accumulator_bpc.iter().sum::<u64>() != f.accumulator_bpc
            || config.activation_read_bpc.iter().sum::<u64>() != f.activation_bpc
            || config.activation_write_bpc.iter().sum::<u64>() != f.activation_bpc
            || config.weight_read_bpc.iter().sum::<u64>() != config.total_weight_read_bpc
            || config.weight_write_bpc.iter().sum::<u64>() != config.total_weight_write_bpc
        {
            return Err("invalid private memory partition or aggregate budget mismatch".into());
        }
        // 16-byte output descriptor: address/identity, K progress and flags.
        // It stays with the FP32 N-vector until the phase drains. Selection and
        // state service use the existing charged control port, not a new oracle.
        let entry = r.n_lanes * 4 + 16;
        let counts: Vec<_> = r
            .jobs
            .iter()
            .map(|j| j.m * j.n.div_ceil(r.n_lanes))
            .collect();
        if counts.iter().sum::<usize>() * entry > f.accumulator_bytes {
            return Err(
                "private output values plus 16-byte descriptors exceed accumulator budget".into(),
            );
        }
        if r.ownership == Ownership::PinnedExpert {
            let mut assigned = vec![0; nc];
            for (j, &count) in counts.iter().enumerate() {
                assigned[pinned[j]] += count * entry;
            }
            if assigned
                .iter()
                .zip(&config.accumulator_bytes)
                .any(|(a, b)| a > b)
            {
                return Err("pinned expert exceeds its private accumulator capacity".into());
            }
        }
        let report = PrivateReport {
            weight_budget_bytes: config
                .weight_slots
                .iter()
                .map(|s| s * r.n_lanes * r.k_lanes * 2)
                .collect(),
            activation_budget_bytes: r
                .m_lanes
                .iter()
                .map(|m| m * f.stages_per_core * r.k_lanes * 2)
                .collect(),
            result_budget_bytes: r
                .m_lanes
                .iter()
                .map(|m| {
                    m * r.result_latency_cycles.div_ceil(r.issue_interval_cycles) as usize
                        * r.n_lanes
                        * 4
                })
                .collect(),
            accumulator_budget_bytes: config.accumulator_bytes.clone(),
            weight_peak_bytes: vec![0; nc],
            activation_peak_bytes: vec![0; nc],
            result_peak_bytes: vec![0; nc],
            accumulator_peak_bytes: vec![0; nc],
            accumulator_metadata_bytes: vec![0; nc],
            owned_output_tiles: vec![0; nc],
            ..Default::default()
        };
        Ok(Self {
            config,
            report,
            weight_read: ports("local_weight_read", nc),
            weight_write: ports("local_weight_write", nc),
            activation_read: ports("local_activation_read", nc),
            activation_write: ports("local_activation_write", nc),
            accumulator: ports("accumulator", nc),
            owners: BTreeMap::new(),
            values: vec![BTreeMap::new(); nc],
        })
    }
    pub fn eligible(&self, r: &Request, key: Key, row: usize, c: usize) -> bool {
        match self.owners.get(&(key.job, key.nt, row)) {
            Some(&owner) => owner == c,
            None => {
                key.kt == 0
                    && self.report.accumulator_peak_bytes[c] + r.n_lanes * 4 + 16
                        <= self.config.accumulator_bytes[c]
            }
        }
    }
    pub fn claim(&mut self, r: &Request, key: Key, row: usize, c: usize) {
        let k = (key.job, key.nt, row);
        if let Some(&owner) = self.owners.get(&k) {
            assert_eq!(owner, c);
            return;
        }
        assert!(self.eligible(r, key, row, c));
        self.owners.insert(k, c);
        self.report.owned_output_tiles[c] += 1;
        self.report.accumulator_peak_bytes[c] += r.n_lanes * 4 + 16;
        self.report.accumulator_metadata_bytes[c] += 16;
        if r.verify_values {
            assert!(self.values[c].insert(k, vec![0.; r.n_lanes]).is_none());
        }
    }
    pub fn accumulate(&mut self, r: &Request, t: &Task, ri: usize, row: usize) {
        let key = (t.key.job, t.key.nt, row);
        assert_eq!(self.owners[&key], t.trace.core);
        if r.verify_values {
            let out = self.values[t.trace.core].get_mut(&key).unwrap();
            for (n, v) in out.iter_mut().take(t.trace.valid_n).enumerate() {
                *v += t.values[ri * t.trace.valid_n + n];
            }
        }
    }
    pub fn gather(&self, r: &Request) -> Option<Vec<Vec<f32>>> {
        if !r.verify_values {
            return None;
        }
        let mut out: Vec<_> = r.jobs.iter().map(|j| vec![0.; j.m * j.n]).collect();
        for bank in &self.values {
            for (&(ji, nt, row), tile) in bank {
                let j = &r.jobs[ji];
                for n in 0..r.n_lanes.min(j.n - nt * r.n_lanes) {
                    out[ji][row * j.n + nt * r.n_lanes + n] = tile[n];
                }
            }
        }
        Some(out)
    }
    pub fn audit(&mut self, r: &Request, f: &Fabric, tasks: &[Task], log: &[Service]) {
        let nc = r.m_lanes.len();
        let expected: usize = r.jobs.iter().map(|j| j.m * j.n.div_ceil(r.n_lanes)).sum();
        assert_eq!(self.owners.len(), expected);
        let mut next: BTreeMap<OutputKey, usize> = BTreeMap::new();
        let mut ordered: Vec<_> = tasks.iter().collect();
        ordered.sort_by_key(|t| t.trace.rmw_done);
        for t in ordered {
            assert!(t.activation.is_none());
            for &row in &t.trace.rows {
                let key = (t.key.job, t.key.nt, row);
                assert_eq!(self.owners[&key], t.trace.core);
                let k = next.entry(key).or_default();
                assert_eq!(*k, t.key.kt);
                *k += 1;
            }
        }
        for (&(j, _, _), &k) in &next {
            assert_eq!(k, r.jobs[j].k.div_ceil(r.k_lanes));
        }
        for c in 0..nc {
            assert!(self.report.accumulator_peak_bytes[c] <= self.config.accumulator_bytes[c]);
            assert!(self.report.weight_peak_bytes[c] <= self.report.weight_budget_bytes[c]);
            assert!(self.report.activation_peak_bytes[c] <= self.report.activation_budget_bytes[c]);
            assert!(self.report.result_peak_bytes[c] <= self.report.result_budget_bytes[c]);
        }
        for s in log
            .iter()
            .filter(|s| s.resource.starts_with("local_") || s.resource == "accumulator")
        {
            self.report
                .local_service_cycles
                .entry(s.resource.clone())
                .or_insert(vec![0; nc])[s.port] += s.end - s.start;
            self.report
                .local_service_bytes
                .entry(s.resource.clone())
                .or_insert(vec![0; nc])[s.port] += s.bytes;
        }
        let mut hash = Sha256::new();
        for (key, c) in &self.owners {
            hash.update(serde_json::to_vec(&(key, c)).unwrap());
        }
        self.report.owner_sha256 = format!("{:x}", hash.finalize());
        self.report.ownership_verified = true;
        // Every invocation performs one activation read and one weight read.
        for resource in ["local_activation_read", "local_weight_read"] {
            assert_eq!(
                log.iter().filter(|s| s.resource == resource).count(),
                tasks.len()
            );
        }
        assert_eq!(
            log.iter()
                .filter(|s| s.resource == "local_activation_write")
                .count(),
            tasks.len()
        );
        assert_eq!(
            self.report.activation_budget_bytes.iter().sum::<usize>(),
            r.m_lanes.iter().sum::<usize>() * f.stages_per_core * r.k_lanes * 2
        );
    }
}
