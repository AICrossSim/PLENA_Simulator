use std::io::Write;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use half::{bf16, f16};
use runtime::Executor;
use sram::{MatrixSram, VectorSram};
use tracing_subscriber::prelude::*;

use crate::cli::{Opts, Parser};
use crate::matrix_machine::MatrixMachine;
use crate::vector_machine::VectorMachine;
use crate::{
    Accelerator, AcceleratorRegFile, BLEN, BROADCAST_AMOUNT, HBM_SIZE, HLEN, MATRIX_SRAM_SIZE,
    MATRIX_SRAM_TYPE, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT, STORE_V_AMOUNT, VECTOR_SRAM_SIZE,
    VECTOR_SRAM_TYPE, VLEN, cli, op,
};

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
        max_loop_instructions = *crate::MAX_LOOP_INSTRUCTIONS,
        "Pipeline"
    );
    tracing::info!(
        settings = %std::env::var("PLENA_SETTINGS_TOML")
            .unwrap_or_else(|_| "default (../plena_settings.toml)".to_string()),
        "Config source"
    );

    let mram = Arc::new(MatrixSram::new(*MLEN, *MATRIX_SRAM_SIZE, *MATRIX_SRAM_TYPE)); // Matrix SRAM
    let vram = Arc::new(VectorSram::from_mx_type(
        *VLEN,
        *VECTOR_SRAM_SIZE,
        *VECTOR_SRAM_TYPE,
    )); // Vector SRAM

    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);

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
    let hbm = Arc::new(memory::WithStats::new(memory::WithTiming::new(
        ManuallyDrop::new(ramulator::Ramulator::hbm2_preset(8).unwrap()),
        memory::MemoryBacked::with_capacity(effective_hbm_size),
    )));

    let mut accelerator = Accelerator {
        m_machine,
        v_machine,
        hbm: hbm.clone(),
        reg_file: AcceleratorRegFile {
            gp_reg: [0; 16],
            fp_reg: [bf16::ZERO; 8],
            hbm_addr_reg: [0; 16],
            scale: 0,
            stride: 1,
            // bmm_scale = 0.25 corresponds to 1/sqrt(head_dim=16).
            // For other head dimensions, the ISA program must set this via
            // the appropriate scalar register instruction before M_BMM/M_BTMM.
            bmm_scale: 0.25,
            v_mask: 0,
        },
        intsram: vec![0; 1024],
        fpsram: vec![bf16::ZERO; 1024],
        loop_stack: Vec::new(),
    };

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
    let fp_vals: Vec<bf16> = {
        let n = fpsram_data.len() / std::mem::size_of::<f16>();
        let f16_slice: &[f16] =
            unsafe { std::slice::from_raw_parts(fpsram_data.as_ptr() as *const f16, n) };
        f16_slice
            .iter()
            .map(|x| bf16::from_f32(f32::from(*x)))
            .collect()
    };

    // Replace the beginning of accelerator.fpsram with fp_vals
    accelerator.fpsram[..fp_vals.len()].copy_from_slice(&fp_vals[..fp_vals.len()]);

    // - INT SRAM Preload
    if let Some(intsram_path) = opts.intsram {
        let intsram_data = std::fs::read(&intsram_path).unwrap_or_else(|err| {
            panic!(
                "failed to read INT SRAM preload file {:?}: {err}",
                intsram_path
            )
        });
        let int_vals: &[u32] = unsafe {
            std::slice::from_raw_parts(
                intsram_data.as_ptr() as *const u32,
                intsram_data.len() / std::mem::size_of::<u32>(),
            )
        };
        accelerator.intsram[..int_vals.len()].copy_from_slice(&int_vals[..int_vals.len()]);
    }
    // - VRAM Preload (if provided)
    if let Some(vram_path) = opts.vram {
        let vram_data = std::fs::read(&vram_path).unwrap_or_else(|err| {
            panic!("failed to read VRAM preload file {:?}: {err}", vram_path)
        });
        accelerator.v_machine.vram.load_from_bytes(&vram_data).await;
    }

    // - Execute Instructions
    // accelerator
    //     .do_ops(&dbg!(
    //         op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>()
    //     ))
    //     .await;
    let decoded_ops = op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>();
    accelerator.do_ops(&decoded_ops).await;

    tracing::debug!("gp1 = {:x}", accelerator.reg_file.gp_reg[1]);
    tracing::debug!("scale = {}", accelerator.reg_file.scale);
    tracing::debug!(
        "Vector SRAM Contents: \n {}",
        accelerator.v_machine.vram.read(0x0000).await.as_tensor()
    );

    tracing::debug!(
        "Matrix SRAM Contents: \n {}",
        accelerator.m_machine.mram.read(0x0000).await.as_tensor()
    );

    tracing::debug!("INT SRAM Contents: \n {:?}", accelerator.intsram);
    tracing::debug!("FP SRAM Contents: \n {:?}", accelerator.fpsram);

    // Dump MRAM
    let mram_bytes = accelerator.m_machine.mram.as_bytes().await;
    dump_to_file("mram_dump.bin", &mram_bytes);

    // Dump VRAM
    let vram_bytes = accelerator.v_machine.vram.as_bytes().await;
    dump_to_file("vram_dump.bin", &vram_bytes);

    // Dump FPSRAM
    let fpsram_bytes: Vec<u8> = accelerator
        .fpsram
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    dump_to_file("fpsram_dump.bin", &fpsram_bytes);

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
