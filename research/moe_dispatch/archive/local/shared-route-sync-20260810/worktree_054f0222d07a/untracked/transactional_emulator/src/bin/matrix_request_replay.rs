#[allow(dead_code)]
#[path = "../load_config.rs"]
mod load_config;
#[allow(dead_code)]
#[path = "../matrix_arch_overlay.rs"]
mod matrix_arch_overlay;
#[allow(dead_code)]
#[path = "../runtime_config.rs"]
mod runtime_config;

use std::path::PathBuf;

use clap::Parser;

use matrix_arch_overlay::ExperimentalMatrixArchitecture;

#[derive(Debug, Parser)]
#[command(about = "Run an isolated PLENA-MoE matrix shortlist request replay")]
struct Opts {
    #[arg(long, value_enum)]
    architecture: ExperimentalMatrixArchitecture,

    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    settings: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    let opts = Opts::parse();
    if let Some(settings) = opts.settings {
        unsafe {
            std::env::set_var("PLENA_SETTINGS_TOML", settings);
        }
    }
    matrix_arch_overlay::evaluate_request_replay_only(
        opts.architecture,
        &opts.manifest,
        &opts.output,
    )
    .await;
}
