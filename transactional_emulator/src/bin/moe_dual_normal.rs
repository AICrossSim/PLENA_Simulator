//! Fixed-route MoE functional execution on independently configured normal cores.
//!
//! This entry point consumes the compiler's versioned job manifest. It keeps
//! the legacy ISA runner intact while sharing the simulator's memory/runtime
//! libraries and one real Ramulator HBM instance across all cores.

#[path = "../moe_normal/mod.rs"]
mod moe_normal;

use std::io::{Read, Write};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::Parser;
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(about = "Execute grouped MoE on normal-buffer cores with shared HBM")]
struct Opts {
    #[arg(long)]
    workload: PathBuf,
    #[arg(long)]
    architecture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Shared HBM2 channel count, identical for baseline and candidate.
    #[arg(long, default_value_t = 8)]
    hbm_channels: u32,
    /// Bound the small experiment's host allocation before loading any image.
    #[arg(long, default_value_t = 536_870_912)]
    max_hbm_bytes: u64,
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

async fn execute(opts: Opts) -> Result<(), Box<dyn std::error::Error>> {
    if !opts.hbm_channels.is_power_of_two() || opts.hbm_channels > 32 {
        return Err("hbm-channels must be a power of two between 1 and 32".into());
    }
    let workload_path = opts.workload.canonicalize()?;
    let architecture_path = opts.architecture.canonicalize()?;
    let workload_bytes = std::fs::read(&workload_path)?;
    let architecture_bytes = std::fs::read(&architecture_path)?;
    let workload_json: serde_json::Value = serde_json::from_slice(&workload_bytes)?;
    let architecture_json: serde_json::Value = serde_json::from_slice(&architecture_bytes)?;
    let image_name = workload_json["hbm_file"]
        .as_str()
        .ok_or("workload.hbm_file must name the encoded weight image")?;
    let image_path = workload_path
        .parent()
        .unwrap()
        .join(image_name)
        .canonicalize()?;
    let image_len = std::fs::metadata(&image_path)?.len();
    if image_len == 0 || image_len % 64 != 0 || image_len > opts.max_hbm_bytes {
        return Err(format!(
            "HBM image must be nonempty, 64-byte aligned and at most {} bytes; got {}",
            opts.max_hbm_bytes, image_len
        )
        .into());
    }
    let output_parent = opts
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let output_name = opts
        .output
        .file_name()
        .ok_or("output requires a filename")?;
    let output_path = output_parent.canonicalize()?.join(output_name);
    let existing_output = output_path
        .canonicalize()
        .unwrap_or_else(|_| output_path.clone());
    if [&workload_path, &architecture_path, &image_path].contains(&&existing_output) {
        return Err("output may not overwrite workload, architecture or HBM input".into());
    }
    let workload: moe_normal::Workload = serde_json::from_slice(&workload_bytes)?;
    let architecture: moe_normal::Architecture = serde_json::from_slice(&architecture_bytes)?;
    moe_normal::validate(&workload, &architecture, image_len).map_err(std::io::Error::other)?;
    let image = std::fs::read(&image_path)?;
    let image_sha256 = format!("{:x}", Sha256::digest(&image));
    if let Some(expected) = workload_json.pointer("/metadata/hbm_sha256")
        && expected.as_str() != Some(image_sha256.as_str())
    {
        return Err("encoded HBM image does not match manifest metadata.hbm_sha256".into());
    }
    let backing = memory::MemoryBacked::with_capacity(usize::try_from(image_len)?);
    backing.with_data(|destination| destination.copy_from_slice(&image));
    drop(image);
    // Match the existing runner's native-model lifetime: the process owns this
    // one HBM model until exit, after the executor drains all memory requests.
    let native = ramulator::Ramulator::hbm2_preset(opts.hbm_channels as usize)?.with_issue_policy(
        architecture
            .dma
            .as_ref()
            .map(|d| d.issue_policy)
            .unwrap_or_default(),
        runtime::Duration::from_picos(architecture.clock_period_ps),
    );
    let native_observer = native.clone();
    let library_path = PathBuf::from(ramulator::raw::Ramulator::library_path()).canonicalize()?;
    let library_sha256 = sha256_file(&library_path)?;
    let hbm: Arc<dyn memory::ErasedMemoryModel> = Arc::new(memory::WithStats::new(
        memory::WithTiming::new(ManuallyDrop::new(native), backing),
    ));
    let report = moe_normal::run(workload, architecture, hbm, image_len)
        .await
        .map_err(std::io::Error::other)?;
    let native_telemetry = native_observer.telemetry();
    if native_telemetry["native_pending"] != 0 {
        return Err("native requests not drained".into());
    }
    if sha256_file(&library_path)? != library_sha256 {
        return Err("native library changed during execution".into());
    }
    let envelope = json!({
        "schema_version": 1,
        "evidence_level": "fixed_route_numerical_moe_with_ramulator_and_analytical_core_timing",
        "provenance": {
            "workload_path": workload_path,
            "workload_sha256": format!("{:x}", Sha256::digest(&workload_bytes)),
            "architecture_path": architecture_path,
            "architecture_sha256": format!("{:x}", Sha256::digest(&architecture_bytes)),
            "hbm_path": image_path,
            "hbm_sha256": image_sha256,
            "hbm_image_bytes": image_len,
            "native_library_path": library_path,
            "native_library_sha256": library_sha256,
            "executable_sha256": sha256_file(&std::env::current_exe()?)?,
        },
        "memory_model": {"name": "Ramulator HBM2 preset", "channels": opts.hbm_channels, "upper_burst_bytes": 64, "calibration": native_telemetry},
        "workload_manifest": workload_json,
        "architecture_manifest": architecture_json,
        "result": report,
    });
    let payload = serde_json::to_vec_pretty(&envelope)?;
    let temporary = output_path.with_file_name(format!(
        ".{}.{}.tmp",
        output_name.to_string_lossy(),
        std::process::id()
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let write_result = (|| -> std::io::Result<()> {
        file.write_all(&payload)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(&temporary, &output_path)
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result?;
    println!("MoE result: {}", output_path.display());
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = execute(Opts::parse()).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
