//! Resource-derived MoE timing. No fixed whole-tile latency/II or hardware claim.
#[allow(dead_code)]
#[path = "../moe_spatial/mod.rs"]
mod moe_spatial;
#[path = "../moe_structural/mod.rs"]
mod structural;
use clap::Parser;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInput {
    name: String,
    m_lanes: Vec<usize>,
    n_tile: usize,
    k_tile: usize,
    jobs: Vec<moe_spatial::Job>,
}

#[derive(Parser)]
struct Opts {
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    hardware: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    operands: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let o = Opts::parse();
    let mut inputs = vec![o.request.clone(), o.hardware.clone()];
    if let Some(p) = &o.operands {
        inputs.extend(moe_spatial::operands::input_paths(p)?);
    }
    for p in inputs {
        if o.output.canonicalize().ok().as_ref() == Some(&p.canonicalize()?) {
            return Err("output cannot overwrite an input".into());
        }
    }
    let input: ModelInput = serde_json::from_slice(&std::fs::read(&o.request)?)?;
    // Adapter only for BF16 verification and native addresses; these legacy
    // timing fields are NOT used by the structural model.
    let request = moe_spatial::Request {
        name: input.name,
        total_multiplier_budget: input.m_lanes.iter().sum::<usize>() * input.n_tile * input.k_tile,
        m_lanes: input.m_lanes,
        n_lanes: input.n_tile,
        k_lanes: input.k_tile,
        result_latency_cycles: 1,
        issue_interval_cycles: 1,
        ownership: moe_spatial::Ownership::TileStealing,
        verify_values: true,
        record_trace: false,
        jobs: input.jobs,
    };
    let hw: structural::Hardware = serde_json::from_slice(&std::fs::read(&o.hardware)?)?;
    let values = o
        .operands
        .as_ref()
        .map(|p| moe_spatial::operands::Operands::load(p, &request))
        .transpose()?;
    let report = structural::run(&request, &hw, values.as_ref())?;
    std::fs::write(o.output, serde_json::to_vec(&report)?)?;
    Ok(())
}
