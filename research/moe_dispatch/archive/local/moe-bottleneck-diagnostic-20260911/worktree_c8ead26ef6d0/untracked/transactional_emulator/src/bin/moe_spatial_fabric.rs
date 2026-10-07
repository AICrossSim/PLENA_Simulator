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
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::parse();
    let path = opts.request.canonicalize()?;
    if opts.output.canonicalize().ok().as_ref() == Some(&path) {
        return Err("output cannot replace request".into());
    }
    let r = serde_json::from_slice(&std::fs::read(path)?)?;
    let report = moe_spatial::fabric::run(&r).map_err(std::io::Error::other)?;
    std::fs::write(opts.output, serde_json::to_vec(&report)?)?;
    Ok(())
}
