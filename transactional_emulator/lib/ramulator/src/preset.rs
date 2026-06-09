use anyhow::Result;

use crate::Ramulator;
use crate::config::Config;

impl Ramulator {
    /// Build a Ramulator from explicit DRAM parameters so the HBM hardware
    /// profile (DRAM standard / bandwidth / channel count / speed) can be
    /// driven from the simulator's TOML config instead of being hard-coded.
    ///
    /// * `dram_impl`     — ramulator2 DRAM standard: "HBM2", "HBM3", "GDDR6", "DDR4", …
    /// * `num_channels`  — channel count; aggregate HBM bandwidth scales ~linearly with it.
    /// * `org_preset`    — ramulator2 organisation preset, e.g. "HBM2_8Gb".
    /// * `timing_preset` — ramulator2 timing/speed preset, e.g. "HBM2_2Gbps".
    ///
    /// `org_preset` / `timing_preset` must be names the linked C++ ramulator2
    /// library recognises for `dram_impl`; an unknown name makes the underlying
    /// `ramulator_new` return null and this returns `Err`.
    pub fn from_params(
        dram_impl: &str,
        num_channels: usize,
        org_preset: &str,
        timing_preset: &str,
    ) -> Result<Self> {
        let config = Config {
            dram: serde_json::json!({
                "impl": dram_impl,
                "org": {
                    "preset": org_preset,
                    "channel": num_channels,
                },
                "timing": {
                    "preset": timing_preset,
                },
            }),
            controller: serde_json::json!({
                "impl": "Generic",
                "Scheduler": {
                    "impl": "FRFCFS",
                },
                "RefreshManager": {
                    "impl": "AllBank",
                },
                "RowPolicy": {
                    "impl": "OpenRowPolicy",
                },
            }),
            addr_mapper: serde_json::json!({
                "impl": "MOP4CLXOR",
            }),
        };
        Self::new(config)
    }

    pub fn ddr4_preset(num_channels: usize) -> Result<Self> {
        Self::from_params("DDR4", num_channels, "DDR4_8Gb_x8", "DDR4_2400R")
    }

    pub fn hbm2_preset(num_channels: usize) -> Result<Self> {
        Self::from_params("HBM2", num_channels, "HBM2_8Gb", "HBM2_2Gbps")
    }
}
