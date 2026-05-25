use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) use clap::Parser;

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

    #[arg(long, short)]
    /// Quiet mode: only output final latency and statistics.
    pub(crate) quiet: bool,

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
}

static QUIET_MODE: AtomicBool = AtomicBool::new(false);

/// Set the global quiet-mode flag (called once at startup from CLI args).
pub(crate) fn set_quiet(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::Relaxed);
}

/// Whether quiet mode is active — suppresses per-instruction logging and
/// non-essential dumps. Read on every hot-path print.
pub(crate) fn is_quiet() -> bool {
    QUIET_MODE.load(Ordering::Relaxed)
}
