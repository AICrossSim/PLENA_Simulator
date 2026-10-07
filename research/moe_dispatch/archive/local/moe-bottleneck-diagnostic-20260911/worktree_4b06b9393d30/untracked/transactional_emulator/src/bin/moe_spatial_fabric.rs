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
    let report = moe_spatial::fabric::run_with_operands(&r, external.as_ref())
        .map_err(std::io::Error::other)?;
    std::fs::write(opts.output, serde_json::to_vec(&report)?)?;
    Ok(())
}
