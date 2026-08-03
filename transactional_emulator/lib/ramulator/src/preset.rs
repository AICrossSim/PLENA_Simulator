use anyhow::Result;

use crate::Ramulator;
use crate::config::{
    AddrMapper, Config, Controller, DDRController, DRAM, RefreshManager, RowPolicy, Scheduler,
    ddr4, hbm2,
};

impl Ramulator {
    pub fn ddr4_preset(num_channels: usize) -> Result<Self> {
        let ddr4 = ddr4::DDR4 {
            timing: ddr4::DDR4Timing::DDR4_2400R,
            org: ddr4::DDR4Org::DDR4_8GB_X8,
        };

        let controller = Controller::GenericDDR(DDRController {
            options: Default::default(),
            scheduler: Scheduler::FrFcFs,
            refresh_manager: RefreshManager::all_bank(),
            row_policy: RowPolicy::Open,
            addr_mapper: AddrMapper::MOP4CLXOR,
            dram: DRAM::DDR4(ddr4.clone()),
        });

        let config = Config {
            controllers: vec![controller; num_channels],
            channel_mapper: Default::default(),
        };
        Self::new(config)
    }

    pub fn hbm2_preset(num_channels: usize) -> Result<Self> {
        let hbm2 = super::config::hbm2::HBM2 {
            timing: hbm2::HBM2Timing::HBM2_2000MBPS,
            org: hbm2::HBM2Org::HBM2_8GB,
        };

        let controller = Controller::HBM12(DDRController {
            options: Default::default(),
            scheduler: Scheduler::FrFcFs,
            refresh_manager: RefreshManager::all_bank(),
            row_policy: RowPolicy::Open,
            addr_mapper: AddrMapper::MOP4CLXOR,
            dram: DRAM::HBM2(hbm2.clone()),
        });

        let config = Config {
            controllers: vec![controller; num_channels],
            channel_mapper: Default::default(),
        };
        Self::new(config)
    }
}
