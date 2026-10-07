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
use std::sync::{Arc, Mutex};

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

// Oracle removes DRAM and its native ingress only. Upstream DMA, decode,
// SRAM, scheduling and numerical memory contents are still exercised.
#[derive(Default, serde::Serialize)]
struct IdealCounters {
    accepted: u64,
    completed: u64,
    pending: u64,
    read_bytes: u64,
    write_bytes: u64,
}

#[derive(Clone, Default)]
struct IdealTiming(Arc<Mutex<IdealCounters>>);
impl IdealTiming {
    fn complete(&self, bytes: u64, write: bool) {
        // Observer state, not hardware storage. Service completes immediately,
        // but still counts the exact 32 B sectors requested by the finite DMA.
        let mut counters = self.0.lock().unwrap();
        counters.accepted += bytes / 32;
        counters.pending += bytes / 32;
        if write {
            counters.write_bytes += bytes;
        } else {
            counters.read_bytes += bytes;
        }
        counters.completed += bytes / 32;
        counters.pending -= bytes / 32;
    }

    fn telemetry(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(&*self.0.lock().unwrap()).unwrap();
        value["request_bytes"] = json!(32);
        value["backend"] = json!("native_timing_bypassed_zero_latency_oracle");
        value
    }
}
impl memory::MemoryTimingModel for IdealTiming {
    async fn read(&self, _: u64) {
        self.complete(64, false);
    }
    async fn read_mask(&self, addr: u64, mask: u8) {
        assert!(addr.is_multiple_of(64) && (1..=3).contains(&mask));
        self.complete(32 * u64::from(mask.count_ones()), false);
    }
    async fn write(&self, _: u64) {
        self.complete(64, true);
    }
    fn supports_sector_reads(&self) -> bool {
        true
    }
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
    let workload: moe_normal::Workload = serde_json::from_slice(&workload_bytes)?;
    let architecture: moe_normal::Architecture = serde_json::from_slice(&architecture_bytes)?;
    let bank = moe_normal::weight_bank::resolve(&workload, &workload_path)
        .map_err(std::io::Error::other)?;
    let image_name = workload_json["hbm_file"]
        .as_str()
        .ok_or("workload.hbm_file must name the encoded weight image")?;
    let image_path = workload_path
        .parent()
        .unwrap()
        .join(image_name)
        .canonicalize()?;
    let image_len = std::fs::metadata(&image_path)?.len();
    if bank.as_ref().is_some_and(|b| b.image_bytes != image_len) {
        return Err("weight bank image length differs from its catalog".into());
    }
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
    if bank
        .as_ref()
        .is_some_and(|b| b.manifest_path == existing_output)
    {
        return Err("output may not overwrite the immutable weight bank catalog".into());
    }
    moe_normal::validate(&workload, &architecture, image_len).map_err(std::io::Error::other)?;
    // Load directly into the backing allocation. A complete expert bank can
    // exceed 512 MiB; avoid a second full image and require an explicit cap.
    let backing = memory::MemoryBacked::with_capacity(usize::try_from(image_len)?);
    let mut image_file = std::fs::File::open(&image_path)?;
    let mut digest = Sha256::new();
    let mut load_result = Ok(());
    backing.with_data(|destination| {
        for chunk in destination.chunks_mut(1024 * 1024) {
            if let Err(error) = image_file.read_exact(chunk) {
                load_result = Err(error);
                return;
            }
            digest.update(chunk);
        }
    });
    load_result?;
    if image_file.read(&mut [0u8; 1])? != 0 {
        return Err("weight image grew during loading".into());
    }
    let image_sha256 = format!("{:x}", digest.finalize());
    if let Some(expected) = workload_json.pointer("/metadata/hbm_sha256")
        && expected.as_str() != Some(image_sha256.as_str())
    {
        return Err("encoded HBM image does not match manifest metadata.hbm_sha256".into());
    }
    if bank
        .as_ref()
        .is_some_and(|b| b.image_sha256 != image_sha256)
    {
        return Err("weight bank encoded image SHA256 mismatch".into());
    }
    // Match the existing runner's native-model lifetime: the process owns this
    // one HBM model until exit, after the executor drains all memory requests.
    let diagnostic = architecture.diagnostic.clone();
    let library_path = PathBuf::from(ramulator::raw::Ramulator::library_path()).canonicalize()?;
    let library_sha256 = sha256_file(&library_path)?;
    let ideal_observer = diagnostic.ideal_hbm.then(IdealTiming::default);
    let (hbm, native_observer): (Arc<dyn memory::ErasedMemoryModel>, _) = if diagnostic.ideal_hbm {
        (
            Arc::new(memory::WithStats::new(memory::WithTiming::new(
                ideal_observer.as_ref().unwrap().clone(),
                backing,
            ))),
            None,
        )
    } else {
        let native = ramulator::Ramulator::hbm2_diagnostic(
            opts.hbm_channels as usize,
            diagnostic.hbm_profile,
        )?
        .with_issue_policy(
            architecture
                .dma
                .as_ref()
                .map(|d| d.issue_policy)
                .unwrap_or_default(),
            runtime::Duration::from_picos(
                architecture
                    .clock_period_ps
                    .div_ceil(diagnostic.dma_speedup),
            ),
        );
        let observer = native.clone();
        (
            Arc::new(memory::WithStats::new(memory::WithTiming::new(
                ManuallyDrop::new(native),
                backing,
            ))),
            Some(observer),
        )
    };
    let report = moe_normal::run(workload, architecture, hbm, image_len)
        .await
        .map_err(std::io::Error::other)?;
    let native_telemetry = native_observer.as_ref().map(|n| n.telemetry());
    let ideal_telemetry = ideal_observer.as_ref().map(IdealTiming::telemetry);
    if let Some(ref observed) = ideal_telemetry
        && (observed["pending"] != 0
            || observed["accepted"] != observed["completed"]
            || observed["read_bytes"] != report.hbm_read_bytes
            || observed["write_bytes"] != report.hbm_write_bytes)
    {
        return Err("oracle requests not drained or sector byte accounting differs".into());
    }
    if native_telemetry
        .as_ref()
        .is_some_and(|t| t["native_pending"] != 0)
    {
        return Err("native requests not drained".into());
    }
    if sha256_file(&library_path)? != library_sha256 {
        return Err("native library changed during execution".into());
    }
    let envelope = json!({
        "schema_version": 1,
        "evidence_level": "fixed_route_numerical_moe_with_explicit_counterfactual_service_diagnostics",
        "diagnostic": diagnostic,
        "provenance": {
            "workload_path": workload_path,
            "workload_sha256": format!("{:x}", Sha256::digest(&workload_bytes)),
            "architecture_path": architecture_path,
            "architecture_sha256": format!("{:x}", Sha256::digest(&architecture_bytes)),
            "hbm_path": image_path,
            "hbm_sha256": image_sha256,
            "hbm_image_bytes": image_len,
            "weight_bank": bank.as_ref().map(|b| json!({
                "manifest_path": b.manifest_path, "manifest_sha256": b.manifest_sha256,
                "state_policy": "cold_start_per_invocation; fixed_physical_weight_image",
            })),
            "native_library_path": library_path,
            "native_library_sha256": library_sha256,
            "executable_sha256": sha256_file(&std::env::current_exe()?)?,
        },
        "memory_model": {"name": if diagnostic.ideal_hbm { "ideal_memory_timing_oracle" } else { "Ramulator HBM2 with explicit timing profile" }, "channels": opts.hbm_channels, "upper_burst_bytes": 64, "calibration": native_telemetry, "oracle": ideal_telemetry},
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

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use memory::{MemoryModel, MemoryTimingModel};

    #[tokio::test]
    async fn ideal_supply_counts_sectors_and_preserves_backing_bytes() {
        let timing = IdealTiming::default();
        let backing = memory::MemoryBacked::with_capacity(64);
        backing.with_data(|bytes| {
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = i as u8;
            }
        });
        let model = memory::WithStats::new(memory::WithTiming::new(timing.clone(), backing));
        for mask in [1, 2, 3] {
            let got = model.read_mask(0, mask).await;
            assert_eq!(got, std::array::from_fn(|i| i as u8));
        }
        timing.read(0).await;
        timing.write(0).await;
        let observed = timing.telemetry();
        assert_eq!(model.statistics().total_bytes_read, 128);
        assert_eq!(observed["read_bytes"], 192);
        assert_eq!(observed["write_bytes"], 64);
        assert_eq!(observed["accepted"], 8);
        assert_eq!(observed["completed"], 8);
        assert_eq!(observed["pending"], 0);
    }
}
