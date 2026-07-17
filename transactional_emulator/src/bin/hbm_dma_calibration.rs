use std::collections::HashSet;
use std::fs;
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant as WallInstant;

use clap::Parser;
use futures::future::join_all;
use memory::{ErasedMemoryModel, MemoryModel, MemoryTimingModel};
use runtime::{Duration, Executor, Instant};
use serde::{Deserialize, Serialize};
use transactional_emulator::dma_calibration::{
    self, DmaFormat, DmaFormatFamily, DmaOpcode, DmaRequestManifest,
    DmaTransfer as TransactionalTransfer,
};

const REQUEST_BYTES: u64 = 64;
const PHYSICAL_BURST_BYTES: u64 = 16;
const DMA_SEMANTIC_VERSION: &str = "production-dma-lines-v2";

#[derive(Parser)]
#[command(about = "Run ordered PLENA DMA streams through Ramulator2 and production dma.rs")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Ramulator2 finalization prints every per-channel counter. Suppress it
    /// for large calibration plans unless explicitly requested for debugging.
    #[arg(long, default_value_t = false)]
    verbose_ramulator_stats: bool,
    /// Reuse completed pattern rows from an existing output file.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Atomically checkpoint the result file after this many new patterns.
    #[arg(long, default_value_t = 25)]
    checkpoint_every: usize,
}

#[cfg(unix)]
fn silence_stdout() {
    unsafe extern "C" {
        fn dup2(oldfd: i32, newfd: i32) -> i32;
    }
    let null = OpenOptions::new().write(true).open("/dev/null").unwrap();
    let result = unsafe { dup2(null.as_raw_fd(), 1) };
    assert!(result >= 0, "failed to redirect Ramulator statistics");
}

#[cfg(not(unix))]
fn silence_stdout() {}

#[derive(Clone, Deserialize)]
struct Plan {
    schema_version: u32,
    patterns: Vec<Pattern>,
}

#[derive(Clone, Deserialize)]
struct Pattern {
    id: String,
    channels: usize,
    repetitions: usize,
    warmup: usize,
    format: HbmFormat,
    transfer: Transfer,
    /// Optional fully expanded heterogeneous production-DMA sequence.
    /// Existing calibration plans leave this empty and continue to expand
    /// ``transfer`` through ``repeat_axes``.
    #[serde(default)]
    transfer_sequence: Vec<Transfer>,
    /// Optional absolute rtl-v1 service-start cycles for ``transfer_sequence``.
    /// The driver advances the event runtime to each target before submitting
    /// that production DMA. If a previous real DMA completes later, the next
    /// transfer starts immediately at the later physical time.
    #[serde(default)]
    transfer_arrival_cycles: Vec<u64>,
    #[serde(default = "default_clock_period_ps")]
    arrival_clock_period_ps: u64,
    #[serde(default)]
    repeat_axes: Vec<RepeatAxis>,
    #[serde(default)]
    conditioner_addresses: Vec<u64>,
    #[serde(default = "default_true")]
    run_transactional: bool,
    #[serde(default = "default_true")]
    run_raw: bool,
    /// Preserve each transfer's completion interval for stateful sequence
    /// diagnostics. Aggregate calibration points keep this disabled.
    #[serde(default)]
    record_occurrence_cycles: bool,
}

fn default_true() -> bool {
    true
}

fn default_clock_period_ps() -> u64 {
    1_000
}

#[derive(Clone, Copy, Deserialize)]
struct HbmFormat {
    #[serde(default = "default_mxfp_family")]
    family: HbmFormatFamily,
    element_bits: u32,
    scale_bits: u32,
    block: u32,
    #[serde(default)]
    exponent_bits: u32,
    #[serde(default)]
    mantissa_bits: u32,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HbmFormatFamily {
    MxInt,
    MxFp,
}

fn default_mxfp_family() -> HbmFormatFamily {
    HbmFormatFamily::MxFp
}

#[derive(Clone, Deserialize)]
struct Transfer {
    opcode: String,
    direction: String,
    #[allow(dead_code)]
    precision: String,
    element_base: u64,
    scale_base: u64,
    dim: u32,
    amount: u32,
    #[serde(alias = "stride")]
    stride_bytes: u32,
    rstride: u8,
    write_amount: u32,
}

#[derive(Clone, Deserialize)]
struct RepeatAxis {
    count: u32,
    #[serde(default, alias = "element_base_delta")]
    element_byte_delta: u64,
    #[serde(default, alias = "scale_base_delta")]
    scale_byte_delta: u64,
}

#[derive(Serialize)]
struct Results<'a> {
    schema_version: u32,
    driver: &'static str,
    dma_semantic_version: &'static str,
    request_manifest_hash_algorithm: &'static str,
    physical_burst_bytes: u64,
    patterns: &'a [PatternResult],
    host_wall_time_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PatternResult {
    id: String,
    raw_ramulator_latency_ns_samples: Option<Vec<f64>>,
    raw_ramulator_median_latency_ns: Option<f64>,
    transactional_dma_latency_ns_samples: Option<Vec<f64>>,
    transactional_dma_median_latency_ns: Option<f64>,
    production_dma_completion_cycles_samples: Option<Vec<u64>>,
    production_dma_median_completion_cycles: Option<f64>,
    #[serde(default)]
    production_dma_occurrence_cycles_samples: Option<Vec<Vec<u64>>>,
    #[serde(default)]
    production_dma_median_occurrence_cycles: Option<Vec<f64>>,
    #[serde(default)]
    production_dma_occurrence_start_cycles_samples: Option<Vec<Vec<u64>>>,
    #[serde(default)]
    production_dma_occurrence_completion_cycles_samples: Option<Vec<Vec<u64>>>,
    #[serde(default)]
    production_dma_median_occurrence_start_cycles: Option<Vec<f64>>,
    #[serde(default)]
    production_dma_median_occurrence_completion_cycles: Option<Vec<f64>>,
    transactional_read_bytes: Option<u64>,
    transactional_write_bytes: Option<u64>,
    request_read_bytes: u64,
    request_write_bytes: u64,
    read_lines: Option<u64>,
    write_lines: Option<u64>,
    full_lines: Option<u64>,
    partial_lines: Option<u64>,
    request_manifest_hash: Option<String>,
    #[serde(default)]
    physical_burst_bytes: Option<u64>,
    #[serde(default)]
    read_physical_bursts: Option<u64>,
    #[serde(default)]
    write_physical_bursts: Option<u64>,
    #[serde(default)]
    read_physical_burst_bytes: Option<u64>,
    #[serde(default)]
    write_physical_burst_bytes: Option<u64>,
}

fn populate_physical_burst_statistics(result: &mut PatternResult) {
    let bursts_per_line = REQUEST_BYTES / PHYSICAL_BURST_BYTES;
    result.physical_burst_bytes = Some(PHYSICAL_BURST_BYTES);
    result.read_physical_bursts = result.read_lines.map(|lines| lines * bursts_per_line);
    result.write_physical_bursts = result.write_lines.map(|lines| lines * bursts_per_line);
    result.read_physical_burst_bytes = result
        .read_physical_bursts
        .map(|bursts| bursts * PHYSICAL_BURST_BYTES);
    result.write_physical_burst_bytes = result
        .write_physical_bursts
        .map(|bursts| bursts * PHYSICAL_BURST_BYTES);
}

#[derive(Deserialize)]
struct ExistingResults {
    schema_version: u32,
    patterns: Vec<PatternResult>,
}

fn write_results(
    path: &PathBuf,
    schema_version: u32,
    patterns: &[PatternResult],
    host_wall_time_seconds: f64,
) {
    let results = Results {
        schema_version,
        driver: if schema_version == 4 {
            "transactional_emulator/hbm_dma_calibration_v4"
        } else {
            "transactional_emulator/hbm_dma_calibration_v3"
        },
        dma_semantic_version: DMA_SEMANTIC_VERSION,
        request_manifest_hash_algorithm: "fnv1a64-v1",
        physical_burst_bytes: PHYSICAL_BURST_BYTES,
        patterns,
        host_wall_time_seconds,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
    fs::rename(temporary, path).unwrap();
}

enum Requests {
    ConcurrentReads(Vec<u64>),
    SequentialReadModifyWrites(Vec<u64>),
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn request_manifest_hash(manifest: &DmaRequestManifest) -> String {
    let mut canonical = format!("{DMA_SEMANTIC_VERSION}\n");
    for address in &manifest.read_lines {
        canonical.push_str(&format!("R:{address:016x}\n"));
    }
    for address in &manifest.write_lines {
        canonical.push_str(&format!("W:{address:016x}\n"));
    }
    format!("fnv1a64:{:016x}", fnv1a64(canonical.as_bytes()))
}

fn blocks_touched(address: u64, length: u64) -> Vec<u64> {
    if length == 0 {
        return Vec::new();
    }
    let mut block = (address / REQUEST_BYTES) * REQUEST_BYTES;
    let end = address + length;
    let mut result = Vec::new();
    while block < end {
        result.push(block);
        block += REQUEST_BYTES;
    }
    result
}

fn request_pattern(transfer: &Transfer, format: HbmFormat) -> Requests {
    assert!(transfer.dim > 0 && transfer.amount > 0);
    assert!((format.element_bits * transfer.dim).is_multiple_of(8));
    let element_len = (format.element_bits * transfer.dim / 8) as u64;
    let scale_len = if format.scale_bits > 0 {
        assert!(transfer.dim.is_multiple_of(format.block));
        let bits = format.scale_bits * (transfer.dim / format.block);
        assert!(bits.is_multiple_of(8));
        (bits / 8) as u64
    } else {
        0
    };
    let stride_bytes = if transfer.rstride == 1 {
        transfer.stride_bytes
    } else {
        format.element_bits * transfer.dim / 8
    };
    let scale_stride_bytes = if format.scale_bits > 0 {
        let logical_stride_bits = stride_bytes * 8;
        assert!(logical_stride_bits.is_multiple_of(format.element_bits));
        let logical_stride = logical_stride_bits / format.element_bits;
        let scale_stride_bits = logical_stride * format.scale_bits / format.block;
        assert!(scale_stride_bits.is_multiple_of(8));
        scale_stride_bits / 8
    } else {
        0
    };

    let mut addresses = Vec::new();
    for index in 0..transfer.amount {
        let element_address = transfer.element_base + (index * stride_bytes) as u64;
        let scale_address = transfer.scale_base + (index * scale_stride_bytes) as u64;
        addresses.extend(blocks_touched(element_address, element_len));
        if scale_len > 0 {
            if transfer.direction == "read" {
                addresses.push((scale_address / REQUEST_BYTES) * REQUEST_BYTES);
            } else {
                addresses.extend(blocks_touched(scale_address, scale_len));
            }
        }
    }
    match transfer.direction.as_str() {
        "read" => Requests::ConcurrentReads(addresses),
        "write" => Requests::SequentialReadModifyWrites(addresses),
        direction => panic!("unsupported DMA direction {direction:?}"),
    }
}

fn expand_stream(pattern: &Pattern) -> Vec<Transfer> {
    if !pattern.transfer_sequence.is_empty() {
        assert!(
            pattern.repeat_axes.is_empty(),
            "transfer_sequence is already expanded and cannot use repeat_axes"
        );
        return pattern.transfer_sequence.clone();
    }

    fn visit(
        axes: &[RepeatAxis],
        axis_index: usize,
        element_delta: u64,
        scale_delta: u64,
        base: &Transfer,
        result: &mut Vec<Transfer>,
    ) {
        if axis_index == axes.len() {
            let mut transfer = base.clone();
            transfer.element_base += element_delta;
            transfer.scale_base += scale_delta;
            result.push(transfer);
            return;
        }
        let axis = &axes[axis_index];
        assert!(axis.count > 0, "repeat axis count must be positive");
        for index in 0..axis.count as u64 {
            visit(
                axes,
                axis_index + 1,
                element_delta + index * axis.element_byte_delta,
                scale_delta + index * axis.scale_byte_delta,
                base,
                result,
            );
        }
    }

    let mut result = Vec::new();
    visit(
        &pattern.repeat_axes,
        0,
        0,
        0,
        &pattern.transfer,
        &mut result,
    );
    result
}

async fn execute_requests(model: &Arc<ramulator::Ramulator>, requests: &Requests) {
    match requests {
        Requests::ConcurrentReads(addresses) => {
            join_all(addresses.iter().map(|address| model.read(*address))).await;
        }
        Requests::SequentialReadModifyWrites(addresses) => {
            for address in addresses {
                model.read(*address).await;
                model.write(*address).await;
            }
        }
    }
}

async fn condition_raw(model: &Arc<ramulator::Ramulator>, addresses: &[u64]) {
    for chunk in addresses.chunks(8) {
        join_all(chunk.iter().map(|address| model.read(*address))).await;
    }
}

async fn execute_raw_stream(model: &Arc<ramulator::Ramulator>, pattern: &Pattern) {
    for transfer in expand_stream(pattern) {
        execute_requests(model, &request_pattern(&transfer, pattern.format)).await;
    }
}

fn request_bytes(pattern: &Pattern) -> (u64, u64) {
    let mut read_requests = 0_u64;
    let mut write_requests = 0_u64;
    for transfer in expand_stream(pattern) {
        match request_pattern(&transfer, pattern.format) {
            Requests::ConcurrentReads(addresses) => {
                read_requests += addresses.len() as u64;
            }
            Requests::SequentialReadModifyWrites(addresses) => {
                read_requests += addresses.len() as u64;
                write_requests += addresses.len() as u64;
            }
        }
    }
    (
        read_requests * REQUEST_BYTES,
        write_requests * REQUEST_BYTES,
    )
}

fn transactional_transfer(
    transfer: &Transfer,
    format: HbmFormat,
) -> Result<TransactionalTransfer, String> {
    let opcode = match transfer.opcode.as_str() {
        "H_PREFETCH_M" => DmaOpcode::PrefetchMatrix,
        "H_PREFETCH_V" => DmaOpcode::PrefetchVector,
        "H_STORE_V" => DmaOpcode::StoreVector,
        opcode => return Err(format!("unsupported DMA opcode {opcode:?}")),
    };
    let family = match format.family {
        HbmFormatFamily::MxInt => DmaFormatFamily::MxInt,
        HbmFormatFamily::MxFp => DmaFormatFamily::MxFp,
    };
    let (exponent_bits, mantissa_bits) = if family == DmaFormatFamily::MxFp {
        if format.exponent_bits == 0 && format.mantissa_bits == 0 {
            match format.element_bits {
                4 => (1, 2),
                8 => (4, 3),
                width => return Err(format!("no default MXFP format for {width} bits")),
            }
        } else {
            (format.exponent_bits, format.mantissa_bits)
        }
    } else {
        (0, 0)
    };
    Ok(TransactionalTransfer {
        opcode,
        element_base: transfer.element_base,
        scale_base: transfer.scale_base,
        dim: transfer.dim,
        amount: transfer.amount,
        stride: transfer.stride_bytes,
        rstride: transfer.rstride,
        write_amount: transfer.write_amount,
        format: DmaFormat {
            family,
            element_bits: format.element_bits,
            scale_bits: format.scale_bits,
            block: format.block,
            exponent_bits,
            mantissa_bits,
        },
    })
}

fn combine_manifest(target: &mut DmaRequestManifest, source: DmaRequestManifest) {
    target.read_lines.extend(source.read_lines);
    target.write_lines.extend(source.write_lines);
    target.full_lines += source.full_lines;
    target.partial_lines += source.partial_lines;
    target.read_bytes += source.read_bytes;
    target.write_bytes += source.write_bytes;
}

fn transactional_stream_manifest(pattern: &Pattern) -> Result<DmaRequestManifest, String> {
    let mut manifest = DmaRequestManifest {
        read_lines: Vec::new(),
        write_lines: Vec::new(),
        full_lines: 0,
        partial_lines: 0,
        read_bytes: 0,
        write_bytes: 0,
    };
    for transfer in expand_stream(pattern) {
        let item =
            dma_calibration::request_manifest(transactional_transfer(&transfer, pattern.format)?)?;
        combine_manifest(&mut manifest, item);
    }
    Ok(manifest)
}

async fn execute_transactional_stream(
    model: &Arc<dyn ErasedMemoryModel>,
    pattern: &Pattern,
) -> Result<(), String> {
    for transfer in expand_stream(pattern) {
        dma_calibration::execute_transactional_dma(
            model.clone(),
            transactional_transfer(&transfer, pattern.format)?,
        )
        .await?;
    }
    Ok(())
}

async fn execute_transactional_stream_profiled(
    model: &Arc<dyn ErasedMemoryModel>,
    pattern: &Pattern,
) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
    let transfers = expand_stream(pattern);
    if !pattern.transfer_arrival_cycles.is_empty()
        && pattern.transfer_arrival_cycles.len() != transfers.len()
    {
        return Err(format!(
            "transfer_arrival_cycles has {} entries for {} transfers",
            pattern.transfer_arrival_cycles.len(),
            transfers.len()
        ));
    }
    if pattern.arrival_clock_period_ps == 0 {
        return Err("arrival_clock_period_ps must be positive".to_string());
    }
    let mut cycles = Vec::new();
    let mut starts = Vec::new();
    let mut completions = Vec::new();
    for (index, transfer) in transfers.into_iter().enumerate() {
        if let Some(target_cycle) = pattern.transfer_arrival_cycles.get(index) {
            let target = Instant::INIT
                + Duration::from_picos(
                    target_cycle.saturating_mul(pattern.arrival_clock_period_ps),
                );
            if Executor::current().now() < target {
                Executor::current().resolve_at(target).await;
            }
        }
        let start = Executor::current().now();
        dma_calibration::execute_transactional_dma(
            model.clone(),
            transactional_transfer(&transfer, pattern.format)?,
        )
        .await?;
        let completion = Executor::current().now();
        let elapsed_picos = (completion - start).as_picos();
        cycles.push(
            elapsed_picos
                .div_ceil(pattern.arrival_clock_period_ps)
                .max(1),
        );
        starts.push(
            (start - Instant::INIT)
                .as_picos()
                .div_ceil(pattern.arrival_clock_period_ps),
        );
        completions.push(
            (completion - Instant::INIT)
                .as_picos()
                .div_ceil(pattern.arrival_clock_period_ps),
        );
    }
    Ok((cycles, starts, completions))
}

async fn measure_raw_once(pattern: Pattern) -> f64 {
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        let model = Arc::new(ramulator::Ramulator::hbm2_preset(pattern.channels).unwrap());
        condition_raw(&model, &pattern.conditioner_addresses).await;
        for _ in 0..pattern.warmup {
            execute_raw_stream(&model, &pattern).await;
        }
        let start = Executor::current().now();
        execute_raw_stream(&model, &pattern).await;
        *task_result.lock().unwrap() = Some((Executor::current().now() - start).as_nanos_f64());
    });
    executor.enter(Instant::ETERNITY).await;
    Arc::try_unwrap(result)
        .unwrap()
        .into_inner()
        .unwrap()
        .unwrap()
}

async fn measure_transactional_once(pattern: Pattern) -> (f64, u64, u64) {
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        let model = Arc::new(memory::WithStats::new(memory::WithTiming::new(
            ramulator::Ramulator::hbm2_preset(pattern.channels).unwrap(),
            memory::NoData,
        )));
        for chunk in pattern.conditioner_addresses.chunks(8) {
            join_all(chunk.iter().map(|address| model.read(*address))).await;
        }
        let erased: Arc<dyn ErasedMemoryModel> = model.clone();
        for _ in 0..pattern.warmup {
            execute_transactional_stream(&erased, &pattern)
                .await
                .unwrap();
        }
        let before = model.statistics();
        let start = Executor::current().now();
        execute_transactional_stream(&erased, &pattern)
            .await
            .unwrap();
        let latency = (Executor::current().now() - start).as_nanos_f64();
        let after = model.statistics();
        *task_result.lock().unwrap() = Some((
            latency,
            after.total_bytes_read - before.total_bytes_read,
            after.total_bytes_written - before.total_bytes_written,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    Arc::try_unwrap(result)
        .unwrap()
        .into_inner()
        .unwrap()
        .unwrap()
}

async fn measure_transactional_timing_once(
    pattern: Pattern,
) -> (
    f64,
    u64,
    u64,
    DmaRequestManifest,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
) {
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        let model = Arc::new(memory::WithStats::new(memory::WithTiming::new(
            ramulator::Ramulator::hbm2_preset(pattern.channels).unwrap(),
            memory::NoData,
        )));
        for chunk in pattern.conditioner_addresses.chunks(8) {
            join_all(chunk.iter().map(|address| model.read(*address))).await;
        }
        let erased: Arc<dyn ErasedMemoryModel> = model.clone();
        for _ in 0..pattern.warmup {
            execute_transactional_stream(&erased, &pattern)
                .await
                .unwrap();
        }
        // Build the canonical request manifest outside the timed interval.
        // V4 uses it for Rust/Python request parity, but the latency target
        // must come from the same production gather/scatter plus delayed-SRAM
        // path used by dispatch.rs, not from the line-only timing helper.
        let manifest = transactional_stream_manifest(&pattern).unwrap();
        let before = model.statistics();
        let start = Executor::current().now();
        let (occurrence_cycles, occurrence_starts, occurrence_completions) =
            if pattern.record_occurrence_cycles {
                execute_transactional_stream_profiled(&erased, &pattern)
                    .await
                    .unwrap()
            } else {
                execute_transactional_stream(&erased, &pattern)
                    .await
                    .unwrap();
                (Vec::new(), Vec::new(), Vec::new())
            };
        let latency = (Executor::current().now() - start).as_nanos_f64();
        let after = model.statistics();
        *task_result.lock().unwrap() = Some((
            latency,
            after.total_bytes_read - before.total_bytes_read,
            after.total_bytes_written - before.total_bytes_written,
            manifest,
            occurrence_cycles,
            occurrence_starts,
            occurrence_completions,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    Arc::try_unwrap(result)
        .unwrap()
        .into_inner()
        .unwrap()
        .unwrap()
}

async fn run_pattern(pattern: Pattern) -> PatternResult {
    let mut raw_samples = Vec::with_capacity(pattern.repetitions);
    let mut transactional_samples = Vec::with_capacity(pattern.repetitions);
    let mut read_bytes = None;
    let mut write_bytes = None;
    let mut request_manifest = None;
    let mut occurrence_samples = Vec::new();
    let mut occurrence_start_samples = Vec::new();
    let mut occurrence_completion_samples = Vec::new();
    for _ in 0..pattern.repetitions {
        if pattern.run_raw {
            raw_samples.push(measure_raw_once(pattern.clone()).await);
        }
        if pattern.run_transactional {
            let (latency, reads, writes, manifest) = if pattern.run_raw {
                let (latency, reads, writes) = measure_transactional_once(pattern.clone()).await;
                (latency, reads, writes, None)
            } else {
                let (
                    latency,
                    reads,
                    writes,
                    manifest,
                    occurrences,
                    occurrence_starts,
                    occurrence_completions,
                ) = measure_transactional_timing_once(pattern.clone()).await;
                if pattern.record_occurrence_cycles {
                    occurrence_samples.push(occurrences);
                    occurrence_start_samples.push(occurrence_starts);
                    occurrence_completion_samples.push(occurrence_completions);
                }
                (latency, reads, writes, Some(manifest))
            };
            transactional_samples.push(latency);
            assert!(read_bytes.is_none_or(|value| value == reads));
            assert!(write_bytes.is_none_or(|value| value == writes));
            if let Some(manifest) = manifest {
                assert!(
                    request_manifest
                        .as_ref()
                        .is_none_or(|value| value == &manifest)
                );
                request_manifest = Some(manifest);
            }
            read_bytes = Some(reads);
            write_bytes = Some(writes);
        }
    }
    let legacy_request_bytes = pattern.run_raw.then(|| request_bytes(&pattern));
    let request_read_bytes = legacy_request_bytes
        .map(|value| value.0)
        .or_else(|| {
            request_manifest
                .as_ref()
                .map(|manifest| manifest.read_bytes)
        })
        .unwrap_or(0);
    let request_write_bytes = legacy_request_bytes
        .map(|value| value.1)
        .or_else(|| {
            request_manifest
                .as_ref()
                .map(|manifest| manifest.write_bytes)
        })
        .unwrap_or(0);
    let median_occurrence_cycles = if pattern.record_occurrence_cycles {
        let count = occurrence_samples.first().map_or(0, Vec::len);
        assert!(
            occurrence_samples
                .iter()
                .all(|sample| sample.len() == count),
            "occurrence timing sample lengths differ across repetitions"
        );
        Some(
            (0..count)
                .map(|index| {
                    median(
                        &occurrence_samples
                            .iter()
                            .map(|sample| sample[index] as f64)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let median_occurrence_values = |samples: &[Vec<u64>]| -> Option<Vec<f64>> {
        if !pattern.record_occurrence_cycles {
            return None;
        }
        let count = samples.first().map_or(0, Vec::len);
        assert!(samples.iter().all(|sample| sample.len() == count));
        Some(
            (0..count)
                .map(|index| {
                    median(
                        &samples
                            .iter()
                            .map(|sample| sample[index] as f64)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
    };
    let mut result = PatternResult {
        id: pattern.id,
        raw_ramulator_median_latency_ns: pattern.run_raw.then(|| median(&raw_samples)),
        raw_ramulator_latency_ns_samples: pattern.run_raw.then_some(raw_samples),
        transactional_dma_median_latency_ns: pattern
            .run_transactional
            .then(|| median(&transactional_samples)),
        production_dma_median_completion_cycles: (!pattern.run_raw && pattern.run_transactional)
            .then(|| median(&transactional_samples)),
        production_dma_completion_cycles_samples: (!pattern.run_raw && pattern.run_transactional)
            .then(|| {
                transactional_samples
                    .iter()
                    .map(|value| value.round() as u64)
                    .collect()
            }),
        production_dma_occurrence_cycles_samples: pattern
            .record_occurrence_cycles
            .then_some(occurrence_samples),
        production_dma_median_occurrence_cycles: median_occurrence_cycles,
        production_dma_occurrence_start_cycles_samples: pattern
            .record_occurrence_cycles
            .then_some(occurrence_start_samples.clone()),
        production_dma_occurrence_completion_cycles_samples: pattern
            .record_occurrence_cycles
            .then_some(occurrence_completion_samples.clone()),
        production_dma_median_occurrence_start_cycles: median_occurrence_values(
            &occurrence_start_samples,
        ),
        production_dma_median_occurrence_completion_cycles: median_occurrence_values(
            &occurrence_completion_samples,
        ),
        transactional_dma_latency_ns_samples: pattern
            .run_transactional
            .then_some(transactional_samples),
        transactional_read_bytes: read_bytes,
        transactional_write_bytes: write_bytes,
        request_read_bytes,
        request_write_bytes,
        read_lines: request_manifest
            .as_ref()
            .map(|value| value.read_lines.len() as u64),
        write_lines: request_manifest
            .as_ref()
            .map(|value| value.write_lines.len() as u64),
        full_lines: request_manifest.as_ref().map(|value| value.full_lines),
        partial_lines: request_manifest.as_ref().map(|value| value.partial_lines),
        request_manifest_hash: request_manifest.as_ref().map(request_manifest_hash),
        physical_burst_bytes: None,
        read_physical_bursts: None,
        write_physical_bursts: None,
        read_physical_burst_bytes: None,
        write_physical_burst_bytes: None,
    };
    populate_physical_burst_statistics(&mut result);
    result
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();
    if !args.verbose_ramulator_stats {
        silence_stdout();
    }
    let plan: Plan = serde_json::from_slice(&fs::read(&args.input).unwrap()).unwrap();
    assert!(
        matches!(plan.schema_version, 2 | 3 | 4),
        "unsupported calibration plan schema"
    );
    assert!(
        !plan.patterns.is_empty(),
        "calibration plan has no patterns"
    );
    assert!(
        args.checkpoint_every > 0,
        "checkpoint_every must be positive"
    );

    let wall_start = WallInstant::now();
    let total_patterns = plan.patterns.len();
    let mut patterns = if args.resume && args.output.exists() {
        let existing: ExistingResults =
            serde_json::from_slice(&fs::read(&args.output).unwrap()).unwrap();
        assert_eq!(
            existing.schema_version, plan.schema_version,
            "resume output schema differs from plan"
        );
        existing.patterns
    } else {
        Vec::new()
    };
    // Older schema-4 checkpoints predate explicit native-burst fields.  They
    // can be enriched exactly from the canonical 64-byte line manifest, so a
    // resumed run never needs to repeat expensive Ramulator measurements.
    for pattern in &mut patterns {
        populate_physical_burst_statistics(pattern);
    }
    let mut completed: HashSet<String> =
        patterns.iter().map(|pattern| pattern.id.clone()).collect();
    let mut since_checkpoint = 0_usize;
    for pattern in plan.patterns {
        assert!(pattern.repetitions > 0, "repetitions must be positive");
        if completed.contains(&pattern.id) {
            continue;
        }
        let result = run_pattern(pattern).await;
        completed.insert(result.id.clone());
        patterns.push(result);
        since_checkpoint += 1;
        if since_checkpoint >= args.checkpoint_every {
            patterns.sort_by(|left, right| left.id.cmp(&right.id));
            write_results(
                &args.output,
                plan.schema_version,
                &patterns,
                wall_start.elapsed().as_secs_f64(),
            );
            eprintln!(
                "HBM DMA calibration: {}/{} patterns complete",
                patterns.len(),
                total_patterns
            );
            since_checkpoint = 0;
        }
    }
    patterns.sort_by(|left, right| left.id.cmp(&right.id));
    write_results(
        &args.output,
        plan.schema_version,
        &patterns,
        wall_start.elapsed().as_secs_f64(),
    );
    eprintln!(
        "HBM DMA calibration complete: {}/{} patterns",
        patterns.len(),
        total_patterns
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer(direction: &str, element_base: u64, scale_base: u64) -> Transfer {
        Transfer {
            opcode: if direction == "read" {
                "H_PREFETCH_M".to_owned()
            } else {
                "H_STORE_V".to_owned()
            },
            direction: direction.to_owned(),
            precision: "weight".to_owned(),
            element_base,
            scale_base,
            dim: 128,
            amount: 128,
            stride_bytes: 64,
            rstride: 1,
            write_amount: 128,
        }
    }

    #[test]
    fn mxint4_prefetch_counts_element_and_scale_requests() {
        let format = HbmFormat {
            family: HbmFormatFamily::MxInt,
            element_bits: 4,
            scale_bits: 8,
            block: 64,
            exponent_bits: 0,
            mantissa_bits: 0,
        };
        let Requests::ConcurrentReads(addresses) =
            request_pattern(&transfer("read", 0, 1 << 20), format)
        else {
            panic!("prefetch must produce reads");
        };
        assert_eq!(addresses.len(), 256);
    }

    #[test]
    fn unaligned_mxint4_store_counts_read_modify_writes() {
        let format = HbmFormat {
            family: HbmFormatFamily::MxInt,
            element_bits: 4,
            scale_bits: 8,
            block: 64,
            exponent_bits: 0,
            mantissa_bits: 0,
        };
        let Requests::SequentialReadModifyWrites(addresses) =
            request_pattern(&transfer("write", 32, (1 << 20) + 32), format)
        else {
            panic!("store must produce read-modify-writes");
        };
        assert_eq!(addresses.len(), 384);
    }

    fn production_manifest(
        direction: &str,
        element_base: u64,
        scale_base: u64,
        dim: u32,
        amount: u32,
    ) -> DmaRequestManifest {
        let transfer = Transfer {
            opcode: if direction == "read" {
                "H_PREFETCH_V".to_owned()
            } else {
                "H_STORE_V".to_owned()
            },
            direction: direction.to_owned(),
            precision: "activation".to_owned(),
            element_base,
            scale_base,
            dim,
            amount,
            stride_bytes: dim / 2,
            rstride: 1,
            write_amount: 1,
        };
        transactional_transfer(
            &transfer,
            HbmFormat {
                family: HbmFormatFamily::MxInt,
                element_bits: 4,
                scale_bits: 8,
                block: 64,
                exponent_bits: 0,
                mantissa_bits: 0,
            },
        )
        .and_then(dma_calibration::request_manifest)
        .unwrap()
    }

    #[test]
    fn production_full_line_store_skips_read() {
        let manifest = production_manifest("write", 0, 60, 128, 1);
        assert!(manifest.read_lines.is_empty());
        assert_eq!(manifest.write_lines, vec![0]);
        assert_eq!(manifest.full_lines, 1);
        assert_eq!(manifest.partial_lines, 0);
    }

    #[test]
    fn production_partial_line_store_reads_once_then_writes_once() {
        let manifest = production_manifest("write", 32, (1 << 20) + 32, 64, 1);
        assert_eq!(manifest.read_lines.len(), 2);
        assert_eq!(manifest.write_lines.len(), 2);
        assert_eq!(manifest.partial_lines, 2);
    }

    #[test]
    fn production_overlapping_element_and_scale_fragments_coalesce() {
        let manifest = production_manifest("write", 0, 16, 64, 1);
        assert_eq!(manifest.read_lines, vec![0]);
        assert_eq!(manifest.write_lines, vec![0]);
        assert_eq!(manifest.partial_lines, 1);
    }
}
