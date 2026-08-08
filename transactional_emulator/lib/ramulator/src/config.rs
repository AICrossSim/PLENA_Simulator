pub mod ddr4;
pub mod hbm2;

#[derive(Debug, serde::Serialize)]
pub struct Config {
    pub controllers: Vec<Controller>,
    pub channel_mapper: ChannelMapper,
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
pub enum ChannelMapper {
    CacheLineInterleave {
        #[serde(skip_serializing_if = "Option::is_none")]
        interleave_bits: Option<u32>,
    },
    Gem5PortInterleave {
        interleave_bits: u32,
    },
}

impl Default for ChannelMapper {
    fn default() -> Self {
        Self::CacheLineInterleave {
            interleave_bits: None,
        }
    }
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
#[derive(Default)]
pub enum Scheduler {
    #[serde(rename = "FRFCFS")]
    #[default]
    FrFcFs,
    #[serde(rename = "FRFCFS-RowHit")]
    FrFcFsRowHit,
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
pub enum RefreshManager {
    AllBank {
        #[serde(skip_serializing_if = "Option::is_none")]
        scatter_interval: Option<u32>,
    },
    HBM34PerBankRefresh,
    NoRefresh,
    PerBank,
}
impl RefreshManager {
    pub fn all_bank() -> RefreshManager {
        RefreshManager::AllBank {
            scatter_interval: None,
        }
    }
}

impl Default for RefreshManager {
    fn default() -> Self {
        Self::AllBank {
            scatter_interval: None,
        }
    }
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
#[derive(Default)]
pub enum RowPolicy {
    #[default]
    Open,
    ClosedCAP {
        #[serde(skip_serializing_if = "Option::is_none")]
        cap: Option<u32>,
    },
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
#[derive(Default)]
pub enum AddrMapper {
    ChRaBaRoCo,
    RoBaRaCoCh,
    RITAddrMapper {
        reserved_rows_per_bank: Option<u32>,
        addr_mapper: Box<AddrMapper>,
    },
    #[default]
    MOP4CLXOR,
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
pub enum DRAM {
    DDR4(ddr4::DDR4),
    HBM2(hbm2::HBM2),
}

#[derive(Debug, serde::Serialize, Default, Clone)]
pub struct DDRControllerOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wr_low_watermark: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wr_high_watermark: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_buffer_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_buffer_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority_buffer_size: Option<u32>,
    // TODO: controller_plugins
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct DDRController {
    #[serde(flatten)]
    pub options: DDRControllerOptions,
    pub scheduler: Scheduler,
    pub refresh_manager: RefreshManager,
    pub row_policy: RowPolicy,
    pub addr_mapper: AddrMapper,
    pub dram: DRAM,
}

#[derive(Debug, serde::Serialize, Clone)]
#[serde(tag = "impl")]
pub enum Controller {
    GenericDDR(DDRController),
    HBM12(DDRController),
}
