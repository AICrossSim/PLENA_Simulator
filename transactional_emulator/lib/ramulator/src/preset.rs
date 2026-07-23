use anyhow::Result;

use crate::Ramulator;
use crate::config::Config;

impl Ramulator {
    pub fn ddr4_preset(num_channels: usize) -> Result<Self> {
        let config = Config {
            dram: serde_json::json!({
                "impl": "DDR4",
                "org": {
                    "preset": "DDR4_8Gb_x8",
                    "channel": num_channels,
                },
                "timing": {
                    "preset": "DDR4_2400R",
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

    pub fn hbm2_preset(num_channels: usize) -> Result<Self> {
        let config = Config {
            dram: serde_json::json!({
                "impl": "HBM2",
                "org": {
                    "preset": "HBM2_8Gb",
                    "channel": num_channels,
                },
                "timing": {
                    "preset": "HBM2_2Gbps",
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

    /// HBM3 preset: same shape as [`Ramulator::hbm2_preset`] with the HBM3
    /// device organization. This ramulator2 build only ships the `HBM3_2Gbps`
    /// timing preset, so per-pin data rate still matches HBM2 — switching
    /// generation changes bank/channel organization and timings only. Real
    /// HBM3 per-pin rates are handled by the analytic model.
    pub fn hbm3_preset(num_channels: usize) -> Result<Self> {
        let config = Config {
            dram: serde_json::json!({
                "impl": "HBM3",
                "org": {
                    "preset": "HBM3_8Gb",
                    "channel": num_channels,
                },
                "timing": {
                    "preset": "HBM3_2Gbps",
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
}
