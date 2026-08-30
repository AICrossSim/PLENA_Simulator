use std::io::Write;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use runtime::{Executor, Instant};
use sram::{MatrixSram, VectorSram};
use tracing_subscriber::prelude::*;

use crate::accelerator::{Accelerator, Scoreboard, TimingDriver, Unit};
use crate::cli::{Opts, Parser, TimingModelOpt};
use crate::matrix_core::MatrixCoreProfile;
use crate::matrix_machine::MatrixMachine;
use crate::runtime_config::{
    BLEN, BROADCAST_AMOUNT, HBM_SIZE, HLEN, MATRIX_SRAM_SIZE, MATRIX_SRAM_TYPE,
    MAX_LOOP_INSTRUCTIONS, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT, STORE_V_AMOUNT,
    VECTOR_SRAM_SIZE, VECTOR_SRAM_TYPE, VLEN,
};
use crate::stage_profile::StageProfiler;
use crate::vector_machine::VectorMachine;
use crate::{cli, op};

/// Write `bytes` to `path` as a diagnostic dump.
///
/// Dumps are post-run artifacts, so a write failure (e.g. read-only cwd, full
/// disk) is logged as a warning and the run continues rather than panicking and
/// discarding an already-completed simulation.
fn dump_to_file(path: &str, bytes: &[u8]) {
    match std::fs::File::create(path).and_then(|mut f| f.write_all(bytes)) {
        Ok(()) => tracing::info!(path, bytes = bytes.len(), "dumped content"),
        Err(err) => tracing::warn!(path, %err, "failed to write dump file"),
    }
}

pub(crate) async fn run_from_cli() {
    let opts = Opts::parse();

    // If --settings is given, set PLENA_SETTINGS_TOML env var BEFORE any
    // LazyLock access (which triggers load_config()). This ensures the
    // per-build TOML is used for all config values.
    if let Some(ref settings_path) = opts.settings {
        // SAFETY: set_var is called before any threads are spawned and before
        // LazyLock statics are accessed, so no concurrent readers exist.
        unsafe { std::env::set_var("PLENA_SETTINGS_TOML", settings_path.as_os_str()) };
    }

    // Initialize tracing subscriber.
    //
    // Filter precedence: `--log-level` (full override) > `RUST_LOG` > default (debug).
    // Output: stderr by default; if `--log-file` is given, also writes to that
    // file (non-blocking appender, no ANSI codes in file).
    let env_filter: tracing_subscriber::EnvFilter = match opts.log_level {
        Some(level) => tracing_subscriber::EnvFilter::new(level.as_level_filter().to_string()),
        None => tracing_subscriber::EnvFilter::builder()
            .with_default_directive(tracing_subscriber::filter::LevelFilter::DEBUG.into())
            .from_env_lossy(),
    };

    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    // Hold the worker guard for the rest of `run_from_cli()` so the appender's
    // background thread isn't dropped before logs are flushed.
    let (file_layer, _file_guard) = match opts.log_file.as_ref() {
        Some(path) => {
            let target = cli::validate_log_file_path(path).unwrap_or_else(|err| {
                // Bootstrap error: the tracing subscriber is not installed yet
                // (we are still building its layers below), so write to stderr.
                eprintln!("error: {}", err);
                std::process::exit(1);
            });
            let appender = tracing_appender::rolling::never(&target.parent, &target.filename);
            let (non_blocking, guard) = tracing_appender::non_blocking(appender);
            let layer = tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false);
            (Some(layer), Some(guard))
        }
        None => (None, None),
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stderr_layer)
        .with(file_layer)
        .init();

    tracing::warn!(
        mlen = *MLEN,
        vlen = *VLEN,
        hlen = *HLEN,
        blen = *BLEN,
        broadcast_amount = *BROADCAST_AMOUNT,
        "Topology"
    );
    tracing::info!(
        matrix_sram_size = *MATRIX_SRAM_SIZE,
        vector_sram_size = *VECTOR_SRAM_SIZE,
        matrix_type = ?*MATRIX_SRAM_TYPE,
        vector_type = ?*VECTOR_SRAM_TYPE,
        "SRAM"
    );
    tracing::info!(
        prefetch_m = *PREFETCH_M_AMOUNT,
        prefetch_v = *PREFETCH_V_AMOUNT,
        store_v = *STORE_V_AMOUNT,
        max_loop_instructions = *MAX_LOOP_INSTRUCTIONS,
        "Pipeline"
    );
    tracing::info!(
        settings = %std::env::var("PLENA_SETTINGS_TOML")
            .unwrap_or_else(|_| "default (../plena_settings.toml)".to_string()),
        "Config source"
    );

    let mram = Arc::new(MatrixSram::new(*MLEN, *MATRIX_SRAM_SIZE, *MATRIX_SRAM_TYPE)); // Matrix SRAM
    let layout_banks = (*VLEN / *BLEN).max(1);
    let vram = Arc::new(VectorSram::from_mx_type_with_banks(
        *VLEN,
        *VECTOR_SRAM_SIZE,
        *VECTOR_SRAM_TYPE,
        layout_banks,
    )); // Vector SRAM

    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let big_core = m_machine.core_profile();
    let tail_core = MatrixCoreProfile::tail_4x256();
    tracing::info!(
        big_core = big_core.name,
        big_rows = big_core.rows,
        big_cols = big_core.cols,
        big_pes = big_core.pe_count(),
        tail_core = tail_core.name,
        tail_rows = tail_core.rows,
        tail_cols = tail_core.cols,
        tail_pes = tail_core.pe_count(),
        tail_pe_area_fraction = tail_core.pe_area_fraction_vs(big_core),
        "Matrix core profiles"
    );

    let v_machine = VectorMachine::new(vram, *VLEN, *HLEN); // Share same dim with VSRAM

    // Allow CLI override of HBM size. The default (from plena_settings.toml)
    // can be 128 GiB to fit large models like LLaDA-8B; tests with smaller
    // preloads should pass --hbm-size to bound the steady-state RSS.
    let effective_hbm_size = opts.hbm_size.unwrap_or(*HBM_SIZE);
    tracing::info!(
        "HBM size: {} bytes ({:.2} GiB)",
        effective_hbm_size,
        effective_hbm_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let dram = ramulator::Ramulator::hbm2_preset(8).unwrap();
    assert_clock_relationship(&dram);
    let hbm = Arc::new(memory::WithStats::new(memory::WithTiming::new(
        ManuallyDrop::new(dram),
        memory::MemoryBacked::with_capacity(effective_hbm_size),
    )));

    let mut accelerator = Accelerator::new(m_machine, v_machine, hbm.clone());

    use std::fs;
    // Panic (rather than exit) on these fatal startup errors so the stack
    // unwinds: that runs the tracing-appender WorkerGuard's Drop, flushing any
    // buffered --log-file output, and preserves the prior exit-101 behavior.
    let op_file = fs::read_to_string(&opts.opcode)
        .unwrap_or_else(|err| panic!("failed to read opcode file {:?}: {err}", opts.opcode));

    let op: Vec<u32> = op_file
        .split_whitespace() // split by spaces/newlines
        .map(|tok| {
            u32::from_str_radix(tok.trim_start_matches("0x"), 16)
                .unwrap_or_else(|err| panic!("failed to parse opcode hex token {tok:?}: {err}"))
        })
        .collect();

    // Memory Initialization
    // - HBM Preload
    let hbm_data = std::fs::read(&opts.hbm)
        .unwrap_or_else(|err| panic!("failed to read HBM preload file {:?}: {err}", opts.hbm));
    hbm.model().data().with_data(|f| {
        f[..hbm_data.len()].copy_from_slice(&hbm_data);
    });

    // Load fpsram and intsram as raw bytes and map to the vector files.
    // - fpsram Preload
    let fpsram_data = std::fs::read(&opts.fpsram).unwrap_or_else(|err| {
        panic!(
            "failed to read FP SRAM preload file {:?}: {err}",
            opts.fpsram
        )
    });
    accelerator.load_fpsram_from_f16_bytes(&fpsram_data);

    // - INT SRAM Preload
    if let Some(intsram_path) = opts.intsram {
        let intsram_data = std::fs::read(&intsram_path).unwrap_or_else(|err| {
            panic!(
                "failed to read INT SRAM preload file {:?}: {err}",
                intsram_path
            )
        });
        accelerator.load_intsram_from_u32_bytes(&intsram_data);
    }
    // - VRAM Preload (if provided)
    if let Some(vram_path) = opts.vram {
        let vram_data = std::fs::read(&vram_path).unwrap_or_else(|err| {
            panic!("failed to read VRAM preload file {:?}: {err}", vram_path)
        });
        accelerator.load_vram_from_bytes(&vram_data).await;
    }

    // - Execute Instructions
    // accelerator
    //     .do_ops(&dbg!(
    //         op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>()
    //     ))
    //     .await;
    let decoded_ops = op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>();
    let mut stage_profiler = opts.stage_profile_asm.as_ref().map(|path| {
        StageProfiler::from_asm(path, decoded_ops.len()).unwrap_or_else(|err| {
            panic!("failed to build stage profile from ASM {:?}: {err}", path)
        })
    });
    let scoreboard_mode = opts.timing_model == TimingModelOpt::Scoreboard;
    crate::timing::set_timing_mode(if scoreboard_mode {
        crate::timing::TimingMode::Scoreboard
    } else {
        crate::timing::TimingMode::Serial
    });
    let mut scoreboard = scoreboard_mode.then(|| {
        let mut sb = Scoreboard::new(opts.scoreboard_serialize);
        if let Some(path) = opts.scoreboard_trace.as_ref() {
            let file = std::fs::File::create(path).unwrap_or_else(|err| {
                panic!("failed to create scoreboard trace file {path:?}: {err}")
            });
            sb.set_trace(Box::new(std::io::BufWriter::new(file)));
        }
        sb
    });
    let timing_driver = match scoreboard.as_mut() {
        Some(scoreboard) => TimingDriver::Scoreboard { scoreboard },
        None => TimingDriver::Serial,
    };
    accelerator
        .do_ops(&decoded_ops, stage_profiler.as_mut(), timing_driver)
        .await;

    let packet = accelerator.lstream_packet_counters();
    tracing::info!(
        packet_reads = packet.read_packets,
        packet_writes = packet.write_packets,
        packet_bank_words = packet.bank_words,
        packet_service_cycles = packet.service_cycles,
        packet_bandwidth_floor_cycles = packet.bandwidth_floor_cycles,
        packet_conflict_stall_cycles = packet.conflict_stall_cycles,
        packet_lane_restore_values = packet.lane_restore_values,
        "L-stream packet counters"
    );

    let serial_duration = Executor::current().now() - Instant::INIT;
    if let Some(sb) = scoreboard.as_ref() {
        let stats = sb.stats;
        let total_picos = serial_duration.as_picos().max(1);
        tracing::info!(
            ops = stats.ops,
            data_stall_picos = stats.data_stall_picos,
            structural_stall_picos = stats.structural_stall_picos,
            dma_wait_picos = stats.dma_wait_picos,
            matrix_busy_pct =
                100.0 * stats.unit_busy_picos[Unit::Matrix.index()] as f64 / total_picos as f64,
            vector_busy_pct =
                100.0 * stats.unit_busy_picos[Unit::Vector.index()] as f64 / total_picos as f64,
            scalar_busy_pct =
                100.0 * stats.unit_busy_picos[Unit::Scalar.index()] as f64 / total_picos as f64,
            dma_busy_pct =
                100.0 * stats.unit_busy_picos[Unit::Dma.index()] as f64 / total_picos as f64,
            matrix_ops = stats.unit_ops[Unit::Matrix.index()],
            vector_ops = stats.unit_ops[Unit::Vector.index()],
            scalar_ops = stats.unit_ops[Unit::Scalar.index()],
            dma_ops = stats.unit_ops[Unit::Dma.index()],
            "Scoreboard timing model summary (total = pipelined cycles)"
        );
    }

    if let Some(profile) = stage_profiler.as_mut() {
        profile.set_total_simulation_duration(serial_duration);
    }

    if let Some(profile) = stage_profiler.as_ref() {
        let out_path = opts
            .stage_profile_out
            .as_deref()
            .unwrap_or_else(|| std::path::Path::new("stage_profile.json"));
        profile
            .write_json(out_path)
            .unwrap_or_else(|err| panic!("failed to write stage profile {:?}: {err}", out_path));
        tracing::info!(path = %out_path.display(), "wrote runtime stage profile");
    }

    accelerator.log_debug_state().await;

    // Dump MRAM
    let mram_bytes = accelerator.mram_dump_bytes().await;
    dump_to_file("mram_dump.bin", &mram_bytes);

    // Dump VRAM
    let vram_bytes = accelerator.vram_dump_bytes().await;
    dump_to_file("vram_dump.bin", &vram_bytes);

    // Dump FPSRAM
    let fpsram_bytes = accelerator.fpsram_dump_bytes();
    dump_to_file("fpsram_dump.bin", &fpsram_bytes);

    // Dump INTSRAM
    let intsram_bytes = accelerator.intsram_dump_bytes();
    dump_to_file("intsram_dump.bin", &intsram_bytes);

    // Dump HBM — skipped unless DEBUG tracing is enabled because HBM_SIZE may
    // be 128 GiB+. Tests run with --log-level warn and don't need hbm_dump.bin;
    // only manual debug runs dump HBM.
    if tracing::enabled!(tracing::Level::DEBUG) {
        let hbm_size = effective_hbm_size;
        let mut hbm_bytes = vec![0u8; hbm_size];
        hbm.model().data().with_data(|f| {
            let len = std::cmp::min(hbm_size, f.len());
            hbm_bytes[..len].copy_from_slice(&f[..len]);
        });
        dump_to_file("hbm_dump.bin", &hbm_bytes);
    }

    let memory_stats = hbm.statistics();
    let utilization = (memory_stats.total_bytes_read + memory_stats.total_bytes_written) as f64
        / Executor::current().now().to_secs();
    tracing::info!(
        "HBM Statistics - Bytes read: {:?} | Bytes written: {:?} | Utilization: {:.2e} bytes/sec",
        memory_stats.total_bytes_read,
        memory_stats.total_bytes_written,
        utilization
    );
}

/// State the accelerator clock and its relationship to the DRAM model's, at
/// startup, instead of letting the two coincide silently.
///
/// `stage_profile.rs` already recorded the hazard: a cycle-domain comparison
/// there "only held because the DRAM tCK happened to equal PERIOD; any preset or
/// frequency change made it fail". Equal is what the HBM2 preset gives -- 2000
/// MBPS is a 1 ns command clock, and `CLOCK_PERIOD_PS` defaults to 1000 -- but
/// that is a property of this preset and this default, not of the design.
///
/// The relationship required is that the DRAM period is a whole multiple of the
/// accelerator period, in either direction. Anything else means a DRAM tick and
/// an accelerator cycle do not line up on any boundary, and every cycle count
/// derived by dividing one by the other is off by a fraction nothing reports.
fn assert_clock_relationship(dram: &ramulator::Ramulator) {
    let accel_ps = crate::runtime_config::PERIOD.as_picos().max(1);
    let dram_ps = dram.period().as_picos().max(1);
    let (hi, lo) = if dram_ps >= accel_ps {
        (dram_ps, accel_ps)
    } else {
        (accel_ps, dram_ps)
    };
    assert!(
        hi % lo == 0,
        "accelerator clock is {accel_ps} ps ({:.3} GHz, from CLOCK_PERIOD_PS) and the \
         DRAM model's is {dram_ps} ps; neither divides the other, so ticks never line \
         up and any cycle count derived from both is off by a fraction nothing reports",
        1e3 / accel_ps as f64,
    );
    println!(
        "Clock: {accel_ps} ps ({:.3} GHz) from CLOCK_PERIOD_PS -- an assumption, not a \
         synthesised frequency. DRAM model: {dram_ps} ps ({}x).",
        1e3 / accel_ps as f64,
        hi / lo,
    );
}
