//! Explicit nonphysical timing experiments. No production defaults are changed.
#[allow(dead_code)]
#[path = "../moe_spatial/mod.rs"]
mod moe_spatial;
use clap::{Parser, ValueEnum};
use moe_spatial::fabric::{Input, WeightRead, WeightSource, run_with_source};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
#[derive(Clone, Copy, ValueEnum)]
enum Source {
    Native,
    IdealOneCycle,
}
#[derive(Parser)]
struct Opts {
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    operands: PathBuf,
    #[arg(long)]
    native_config: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    source: Source,
    #[arg(long)]
    label: String,
}
#[derive(Default, Serialize)]
struct Ideal {
    submitted: usize,
    completed: usize,
    requested_bytes: u64,
    outstanding_peak: usize,
    #[serde(skip)]
    pending: BTreeMap<u64, Vec<usize>>,
    #[serde(skip)]
    ids: BTreeSet<usize>,
}
impl WeightSource for Ideal {
    fn submit(&mut self, r: WeightRead<'_>) -> Result<(), String> {
        if !self.ids.insert(r.id) {
            return Err("duplicate ideal source descriptor".into());
        }
        self.submitted += 1;
        self.requested_bytes += (r.valid_n * r.valid_k * 2).div_ceil(32) as u64 * 32;
        self.outstanding_peak = self.outstanding_peak.max(self.submitted - self.completed);
        self.pending.entry(r.now + 1).or_default().push(r.id);
        Ok(())
    }
    fn advance(&mut self, now: u64) -> Result<Vec<usize>, String> {
        let done = self.pending.remove(&now).unwrap_or_default();
        assert!(self.pending.keys().all(|&t| t > now));
        self.completed += done.len();
        Ok(done)
    }
    fn drained(&self) -> bool {
        self.pending.is_empty() && self.submitted == self.completed
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let o = Opts::parse();
    for p in [&o.request, &o.operands, &o.native_config]
        .into_iter()
        .cloned()
        .chain(moe_spatial::operands::input_paths(&o.operands).map_err(std::io::Error::other)?)
    {
        if o.output.canonicalize().ok().as_ref() == Some(&p.canonicalize()?) {
            return Err("output cannot overwrite inputs".into());
        }
    }
    let r: Input = serde_json::from_slice(&std::fs::read(&o.request)?)?;
    if !r.compute.verify_values || r.fabric.private_memories.is_none() {
        return Err("diagnostic requires private memories and numerical verification".into());
    }
    let data = moe_spatial::operands::Operands::load(&o.operands, &r.compute)
        .map_err(std::io::Error::other)?;
    let cfg: moe_spatial::native::Config =
        serde_json::from_slice(&std::fs::read(&o.native_config)?)?;
    let period = cfg.core_period_ps;
    let (mut result, source_report) = match o.source {
        Source::Native => {
            let mut src =
                moe_spatial::native::Native::new(cfg, &r.compute).map_err(std::io::Error::other)?;
            let report =
                run_with_source(&r, Some(&data), Some(&mut src)).map_err(std::io::Error::other)?;
            (
                report,
                serde_json::json!({"kind":"native_hbm_retained","report":src.report()}),
            )
        }
        Source::IdealOneCycle => {
            let mut src = Ideal::default();
            let report =
                run_with_source(&r, Some(&data), Some(&mut src)).map_err(std::io::Error::other)?;
            assert!(src.drained());
            assert_eq!(src.requested_bytes, report.stats.source_weight_bytes);
            (
                report,
                serde_json::json!({"kind":"oracle_weight_response_one_cycle","drained":src.drained(),"report":src}),
            )
        }
    };
    result.scope = match o.source {
        Source::Native => "ORACLE: native HBM retained; selected control/on-chip service times removed; private capacities unchanged; not layer E2E",
        Source::IdealOneCycle => "ORACLE: nonphysical one-cycle weight response, no native HBM backend; private capacities unchanged; not layer E2E",
    }.into();
    let bytes = serde_json::to_vec(
        &serde_json::json!({"schema":"plena_private_core_oracle_v1","nonphysical":true,
        "label":o.label,"total_time_ps":result.total_cycles*period,"result":result,"source":source_report,
        "scope":"same online policy, not frozen task placement; timing-only interventions preserve finite storage and numerical data; not layer E2E"}),
    )?;
    std::fs::write(o.output, bytes)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ideal_source_responds_after_one_cycle_and_counts_bytes() {
        let j = moe_spatial::Job {
            expert: 0,
            m: 1,
            n: 4,
            k: 512,
            seed: 1,
        };
        let mut src = Ideal::default();
        src.submit(WeightRead {
            id: 0,
            job: &j,
            n_start: 0,
            k_start: 0,
            valid_n: 4,
            valid_k: 512,
            now: 10,
        })
        .unwrap();
        assert!(src.advance(10).unwrap().is_empty());
        assert!(!src.drained());
        assert_eq!(src.advance(11).unwrap(), vec![0]);
        assert!(src.drained());
        assert_eq!(src.requested_bytes, 4096);
        assert!(
            src.submit(WeightRead {
                id: 0,
                job: &j,
                n_start: 0,
                k_start: 0,
                valid_n: 4,
                valid_k: 512,
                now: 12
            })
            .is_err()
        );
    }
}
