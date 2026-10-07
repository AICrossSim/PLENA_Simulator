#[allow(dead_code)]
#[path = "../load_config.rs"]
mod load_config;
#[path = "../mkn_request_replay.rs"]
mod mkn_request_replay;
#[allow(dead_code)]
#[path = "../runtime_config.rs"]
mod runtime_config;

use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Replay an arbitrary PLENA-MoE (M,K,N) shortlist against 4x1024")]
struct Opts {
    #[arg(long)]
    architecture_manifest: PathBuf,

    #[arg(long)]
    workload_manifest: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 3)]
    repeats: usize,

    #[arg(long)]
    baseline_report: Option<PathBuf>,

    #[arg(long)]
    settings: Option<PathBuf>,

    /// Experiment: model the scale-inlined tile-major weight layout
    /// (1.125 B/param) instead of the current separate-scale layout
    /// (2.00 B/param).  Off by default.
    #[arg(long, default_value_t = false)]
    inline_scale_layout: bool,

    /// Experiment: one routed job per (token, slot) pair, reloading expert
    /// weights per pair, i.e. today's compiler behaviour.  Off by default
    /// (grouped: one job per expert).
    #[arg(long, default_value_t = false)]
    pair_major: bool,

    /// Experiment: pipeline macro-tile issue at the Matrix-SRAM/PE boundary.
    /// A core accepts a new tile every max(m,n) cycles instead of being held
    /// for the whole tile latency.  Off by default.
    #[arg(long, default_value_t = false)]
    pipelined_issue: bool,

    /// Experiment: lossless per-tile entropy coding of MX weights in HBM
    /// (measured 1.283x on real weights), decoded before Matrix SRAM.
    /// Requires --inline-scale-layout.  Off by default.
    #[arg(long, default_value_t = false)]
    compressed_tiles: bool,
}

#[tokio::main]
async fn main() {
    let opts = Opts::parse();
    if let Some(settings) = opts.settings {
        unsafe {
            std::env::set_var("PLENA_SETTINGS_TOML", settings);
        }
    }
    if opts.inline_scale_layout {
        unsafe {
            std::env::set_var("PLENA_MKN_INLINE_SCALE_LAYOUT", "1");
        }
    }
    if opts.pair_major {
        unsafe {
            std::env::set_var("PLENA_MKN_PAIR_MAJOR", "1");
        }
    }
    if opts.pipelined_issue {
        unsafe {
            std::env::set_var("PLENA_MKN_PIPELINED_ISSUE", "1");
        }
    }
    if opts.compressed_tiles {
        assert!(
            opts.inline_scale_layout,
            "--compressed-tiles requires --inline-scale-layout"
        );
        unsafe {
            std::env::set_var("PLENA_MKN_COMPRESSED_TILES", "1");
        }
    }
    mkn_request_replay::evaluate(
        &opts.architecture_manifest,
        &opts.workload_manifest,
        &opts.output,
        opts.repeats,
        opts.baseline_report.as_deref(),
    )
    .await;
}
