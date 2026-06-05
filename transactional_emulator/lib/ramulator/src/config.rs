use anyhow::Result;

pub struct HBM2Timing {
    /// Transfer rate in MT/s
    pub rate: u32,

    pub n_bl: u32,
    pub n_cl: u32,
    pub n_rcdrd: u32,
    pub n_rcdwr: u32,
    pub n_rp: u32,
    pub n_ras: u32,
    pub n_rc: u32,
    pub n_wr: u32,
    pub n_rtps: u32,
    pub n_rtpl: u32,
    pub n_cwl: u32,
    pub n_ccds: u32,
    pub n_ccdl: u32,
    pub n_rrds: u32,
    pub n_rrdl: u32,
    pub n_wtrs: u32,
    pub n_wtrl: u32,
    pub n_rtw: u32,
    pub n_faw: u32,
    pub n_rfcsb: u32,
    pub n_refi: u32,
    pub n_rrefd: u32,

    /// tRFC in nanoseconds
    pub t_rfc: u32,
    /// tREFISB in nanoseconds
    pub t_refisb: u32,
}

impl HBM2Timing {
    pub fn preset_8gb_2gbps() -> Self {
        // Numbers from https://github.com/CMU-SAFARI/ramulator2/blob/be93be78055d922aa1d4d33e15bcc8f2b0c61a9d/src/dram/impl/HBM2.cpp#L6
        Self {
            rate: 2000,
            n_bl: 4,
            n_cl: 7,
            n_rcdrd: 7,
            n_rcdwr: 7,
            n_rp: 7,
            n_ras: 17,
            n_rc: 19,
            n_wr: 8,
            n_rtps: 2,
            n_rtpl: 3,
            n_cwl: 2,
            n_ccds: 1,
            n_ccdl: 2,
            n_rrds: 2,
            n_rrdl: 3,
            n_wtrs: 3,
            n_wtrl: 4,
            n_rtw: 3,
            n_faw: 15,
            n_rfcsb: 160,
            n_refi: 3900,
            n_rrefd: 8,
            t_rfc: 350,
            t_refisb: 2438,
        }
    }
}

#[derive(serde::Deserialize)]
pub struct Config {
    #[serde(rename = "DRAM")]
    pub(crate) dram: serde_json::Value,
    #[serde(rename = "Controller")]
    pub(crate) controller: serde_json::Value,
    #[serde(rename = "AddrMapper")]
    pub(crate) addr_mapper: serde_json::Value,
}

impl Config {
    pub fn from_yaml(config: &str) -> Result<Self> {
        Ok(serde_yaml_ng::from_str(config)?)
    }

    pub fn from_hbm2_preset(timing: HBM2Timing, num_channels: u32) -> Self {
        let config = Config {
            dram: serde_json::json!({
                "impl": "HBM2",
                "org": {
                    "preset": "HBM2_8Gb",
                    "channel": num_channels,
                },
                "timing": {
                    "rate": timing.rate,
                    "nBL": timing.n_bl,
                    "nCL": timing.n_cl,
                    "nRCDRD": timing.n_rcdrd,
                    "nRCDWR": timing.n_rcdwr,
                    "nRP": timing.n_rp,
                    "nRAS": timing.n_ras,
                    "nRC": timing.n_rc,
                    "nWR": timing.n_wr,
                    "nRTPS": timing.n_rtps,
                    "nRTPL": timing.n_rtpl,
                    "nCWL": timing.n_cwl,
                    "nCCDS": timing.n_ccds,
                    "nCCDL": timing.n_ccdl,
                    "nRRDS": timing.n_rrds,
                    "nRRDL": timing.n_rrdl,
                    "nWTRS": timing.n_wtrs,
                    "nWTRL": timing.n_wtrl,
                    "nRTW": timing.n_rtw,
                    "nFAW": timing.n_faw,
                    "nRFCSB": timing.n_rfcsb,
                    "nRREFD": timing.n_rrefd,
                    "nREFI": timing.n_refi,
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
        config
    }
}
