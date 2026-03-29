use std::env;
use std::path::{Path, PathBuf};

const LIB_ENV_VARS: [&str; 4] = [
    "RAMULATOR_LIB_DIR",
    "LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
];

const LIB_CANDIDATES: [&str; 3] = ["libramulator.dylib", "libramulator.so", "libramulator.a"];

fn has_ramulator_library(dir: &Path) -> bool {
    LIB_CANDIDATES.iter().any(|name| dir.join(name).exists())
}

fn main() {
    for key in LIB_ENV_VARS {
        println!("cargo:rerun-if-env-changed={key}");
    }

    let mut found_dirs = Vec::<PathBuf>::new();
    for key in LIB_ENV_VARS {
        let Some(value) = env::var_os(key) else {
            continue;
        };

        for dir in env::split_paths(&value) {
            if has_ramulator_library(&dir) && !found_dirs.iter().any(|seen| seen == &dir) {
                println!("cargo:rustc-link-search=native={}", dir.display());
                found_dirs.push(dir);
            }
        }
    }

    if found_dirs.is_empty() {
        println!(
            "cargo:warning=Could not locate libramulator. Set RAMULATOR_LIB_DIR to the directory containing libramulator, or build inside `nix develop` so LIBRARY_PATH/DYLD_LIBRARY_PATH are populated."
        );
    }
}
