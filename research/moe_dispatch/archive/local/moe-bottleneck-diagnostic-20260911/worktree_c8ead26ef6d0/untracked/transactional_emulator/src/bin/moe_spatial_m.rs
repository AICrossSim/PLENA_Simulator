//! Independent spatial-M compute experiment. See doc/moe_spatial_m_contract.md.
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
    let request_path = opts.request.canonicalize()?;
    if opts.output.canonicalize().ok().as_ref() == Some(&request_path) {
        return Err("output must not replace input".into());
    }
    let bytes = std::fs::read(request_path)?;
    let request: moe_spatial::Request = serde_json::from_slice(&bytes)?;
    let result = moe_spatial::run(&request).map_err(std::io::Error::other)?;
    std::fs::write(opts.output, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
