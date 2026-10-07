#[allow(dead_code)]
#[path = "../moe_spatial/mod.rs"]
mod moe_spatial;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Opts {
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Actual BF16 X/W files with shapes and SHA256 enforced against the jobs.
    #[arg(long)]
    operands: Option<PathBuf>,
    /// Opt in to live native HBM2 weight timing (JSON config); old default unchanged.
    #[arg(long)]
    native_weight_config: Option<PathBuf>,
    /// Explicit rate/latency/credit source; never labeled native HBM.
    #[arg(long, conflicts_with = "native_weight_config")]
    analytical_weight_config: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::parse();
    let path = opts.request.canonicalize()?;
    if opts.output.canonicalize().ok().as_ref() == Some(&path) {
        return Err("output cannot replace request".into());
    }
    let r: moe_spatial::fabric::Input = serde_json::from_slice(&std::fs::read(path)?)?;
    let external = opts
        .operands
        .as_ref()
        .map(|p| moe_spatial::operands::Operands::load(p, &r.compute))
        .transpose()
        .map_err(std::io::Error::other)?;
    if let Some(p) = &opts.operands {
        for source in moe_spatial::operands::input_paths(p).map_err(std::io::Error::other)? {
            if opts.output.canonicalize().ok().as_ref() == Some(&source.canonicalize()?) {
                return Err("output cannot replace an operand input".into());
            }
        }
    }
    let payload = if let Some(path) = &opts.analytical_weight_config {
        if opts.output.canonicalize().ok().as_ref() == Some(&path.canonicalize()?) {
            return Err("output cannot replace analytical config".into());
        }
        let cfg: moe_spatial::analytical::Config = serde_json::from_slice(&std::fs::read(path)?)?;
        let mut src =
            moe_spatial::analytical::Analytical::new(cfg).map_err(std::io::Error::other)?;
        let mut report =
            moe_spatial::fabric::run_with_source(&r, external.as_ref(), Some(&mut src))
                .map_err(std::io::Error::other)?;
        report.scope="analytical bandwidth/latency/credit source plus finite cycle execution; BF16 GEMMs only; NOT native HBM or full model".into();
        serde_json::to_vec(
            &serde_json::json!({"schema":"plena_band_prefetch_analytical_v1","core_period_ps":1000,
            "total_time_ps":report.total_cycles*1000,"result":report,"analytical":src.report()}),
        )?
    } else if let Some(path) = &opts.native_weight_config {
        if opts.output.canonicalize().ok().as_ref() == Some(&path.canonicalize()?) {
            return Err("output cannot replace native config".into());
        }
        if r.fabric.zero_weight_time {
            return Err("native backend and zero-weight oracle cannot be combined".into());
        }
        let config: moe_spatial::native::Config = serde_json::from_slice(&std::fs::read(path)?)?;
        let period = config.core_period_ps;
        let mut native =
            moe_spatial::native::Native::new(config, &r.compute).map_err(std::io::Error::other)?;
        let report = moe_spatial::fabric::run_with_source(&r, external.as_ref(), Some(&mut native))
            .map_err(std::io::Error::other)?;
        serde_json::to_vec(
            &serde_json::json!({"schema":"plena_spatial_native_weight_v1",
            "total_time_ps":report.total_cycles * period,"result":report,"native":native.report()}),
        )?
    } else {
        let report = moe_spatial::fabric::run_with_operands(&r, external.as_ref())
            .map_err(std::io::Error::other)?;
        serde_json::to_vec(&report)?
    };
    std::fs::write(opts.output, payload)?;
    Ok(())
}
