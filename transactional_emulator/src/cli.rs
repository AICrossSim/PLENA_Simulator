use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub(crate) use clap::Parser;
use clap::ValueEnum;

/// Log level filter for the tracing subscriber.
///
/// When passed via `--log-level`, this fully overrides the `RUST_LOG`
/// environment variable. Omit `--log-level` to fall back to `RUST_LOG`,
/// and if neither is set the default is `debug`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Off,
}

impl LogLevel {
    pub(crate) fn as_level_filter(self) -> tracing_subscriber::filter::LevelFilter {
        use tracing_subscriber::filter::LevelFilter;
        match self {
            Self::Trace => LevelFilter::TRACE,
            Self::Debug => LevelFilter::DEBUG,
            Self::Info => LevelFilter::INFO,
            Self::Warn => LevelFilter::WARN,
            Self::Error => LevelFilter::ERROR,
            Self::Off => LevelFilter::OFF,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ProfileMemoryLevel {
    Aggregate,
    Opcode,
}

impl ProfileMemoryLevel {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Aggregate => "aggregate",
            Self::Opcode => "opcode",
        }
    }
}

/// Parent directory + filename split of a `--log-file` argument, ready
/// to hand to [`tracing_appender::rolling::never`].
pub(crate) struct LogFileTarget {
    pub(crate) parent: PathBuf,
    pub(crate) filename: OsString,
}

/// Validate and split a `--log-file` path into `(parent, filename)`.
///
/// - If the path has no parent component (or an empty one), the current
///   directory `.` is used as the parent.
/// - Errors if the parent directory does not exist or if the path has no
///   filename component.
pub(crate) fn validate_log_file_path(path: &Path) -> Result<LogFileTarget, String> {
    let parent: PathBuf = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    if !parent.exists() {
        return Err(format!(
            "--log-file parent directory does not exist: {}",
            parent.display()
        ));
    }
    let filename = path
        .file_name()
        .ok_or_else(|| {
            format!(
                "--log-file must include a filename, got: {}",
                path.display()
            )
        })?
        .to_os_string();
    Ok(LogFileTarget { parent, filename })
}

/// Parse a human-readable byte-count string into a `usize` byte value.
///
/// Accepted forms (case-insensitive suffix, optional 'i' for IEC):
///   - no suffix  → bytes  (e.g. `1048576`)
///   - `K` / `KiB` → × 1 024
///   - `M` / `MiB` → × 1 048 576
///   - `G` / `GiB` → × 1 073 741 824
///   - `T` / `TiB` → × 1 099 511 627 776
///
/// Examples: `256M`, `256MiB`, `1G`, `512K`, `1073741824`
fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    // Split at first non-digit character (after an optional leading sign).
    let split_pos = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num_str, suffix) = s.split_at(split_pos);
    let num: usize = num_str
        .parse()
        .map_err(|_| format!("invalid number in size string {:?}", s))?;
    let suffix_upper = suffix.to_uppercase();
    // Strip optional trailing 'IB' or 'B' to normalise KiB→K, KB→K, K→K.
    let key = suffix_upper
        .trim_end_matches("IB")
        .trim_end_matches('B')
        .trim_end_matches('I');
    let mult: usize = match key {
        "" => 1,
        "K" => 1 << 10,
        "M" => 1 << 20,
        "G" => 1 << 30,
        "T" => 1_usize << 40,
        other => return Err(format!("unrecognised size suffix {:?} in {:?}", other, s)),
    };
    num.checked_mul(mult)
        .ok_or_else(|| format!("size {:?} overflows usize", s))
}

#[derive(Parser)]
pub(crate) struct Opts {
    #[arg(long)]
    /// Path to Instruction to be executed.
    pub(crate) opcode: PathBuf,

    #[arg(long)]
    /// Path to HBM contents for preloading.
    pub(crate) hbm: PathBuf,

    #[arg(long)]
    /// Path to FP SRAM contents for preloading.
    pub(crate) fpsram: PathBuf,

    #[arg(long)]
    /// Path to INT SRAM contents for preloading.
    pub(crate) intsram: Option<PathBuf>,

    #[arg(long)]
    /// Path to file storing Vector SRAM contents (optional).
    pub(crate) vram: Option<PathBuf>,

    /// Log level filter (overrides `RUST_LOG` when set). Default: debug.
    #[arg(long, value_enum, help_heading = "Logging")]
    pub(crate) log_level: Option<LogLevel>,

    /// Also write logs to this file (in addition to stderr).
    ///
    /// The parent directory must exist. ANSI colour codes are stripped
    /// when writing to file.
    #[arg(long, help_heading = "Logging")]
    pub(crate) log_file: Option<PathBuf>,

    #[arg(long, value_parser = parse_size)]
    /// Override HBM allocation size (default: from plena_settings.toml).
    ///
    /// Accepts a bare byte count or a human-readable suffix:
    ///   256M / 256MiB   →  268 435 456 bytes
    ///   1G   / 1GiB     →  1 073 741 824 bytes
    ///   512K / 512KiB   →  524 288 bytes
    ///   1073741824      →  raw bytes (legacy form, still accepted)
    ///
    /// The emulator's HBM is `MemoryBacked`, a `Vec<[u8;64]>` of `hbm_size/64`
    /// entries. On Linux this maps to a single `mmap`-backed virtual region;
    /// physical RAM is committed page-by-page as the emulator touches HBM
    /// addresses. With the default 128 GiB setting in `plena_settings.toml`
    /// (sized for LLaDA-8B's full weight set), the steady-state RSS for a
    /// long ASM trace can grow to 100+ GiB even when only a few hundred MiB
    /// of HBM are actually populated, because the test ASM dereferences
    /// addresses spread across the full virtual range. Tests that preload
    /// only a small HBM prefix can pass e.g. `--hbm-size 256M` to bound the
    /// steady-state RSS.
    pub(crate) hbm_size: Option<usize>,

    #[arg(long)]
    /// Override modeled Ramulator HBM channel count.
    ///
    /// Default comes from plena_settings.toml (or 8 for legacy TOMLs). With the
    /// current HBM2_2Gbps, 64-bit/channel model, 128 channels gives an
    /// A100-bandwidth-equivalent peak of 2048 GB/s, but it is not a physical
    /// A100 HBM2e topology.
    pub(crate) hbm_channels: Option<u32>,

    #[arg(long)]
    /// Path to plena_settings.toml. Overrides PLENA_SETTINGS_TOML env var and
    /// the default ../plena_settings.toml lookup.
    pub(crate) settings: Option<PathBuf>,

    #[arg(long)]
    /// Enable simulated latency profiling by opcode/category.
    pub(crate) profile_memory: bool,

    #[arg(long, value_enum, default_value_t = ProfileMemoryLevel::Opcode)]
    /// Profiler detail level. aggregate emits category buckets only; opcode also emits per-opcode buckets.
    pub(crate) profile_memory_level: ProfileMemoryLevel,

    #[arg(long)]
    /// Path to write memory profile JSON. Defaults to memory_profile.json next to --opcode.
    pub(crate) profile_output: Option<PathBuf>,
}
