use anyhow::Result;

use crate::Ramulator;
use crate::config::{
    AddrMapper, Config, Controller, DDRController, DRAM, RefreshManager, RowPolicy, Scheduler,
    ddr4, hbm2, hbm3,
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

    /// HBM2 preset: `HBM2_8Gb` organisation at 2.0 Gb/s per pin, 64-bit
    /// channels, one `HBM12` controller per channel.
    pub fn hbm2_preset(num_channels: usize) -> Result<Self> {
        Self::new(Self::hbm2_config(num_channels))
    }

    /// The configuration behind [`Ramulator::hbm2_preset`].
    pub fn hbm2_config(num_channels: usize) -> Config {
        let hbm2 = hbm2::HBM2 {
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

        Config {
            controllers: vec![controller; num_channels],
            channel_mapper: Default::default(),
        }
    }

    /// HBM3 preset: the controller policies of [`Ramulator::hbm2_preset`]
    /// with the HBM3 device model on the dual-command-bus `HBM34` controller
    /// upstream pairs it with. `HBM3_8Gb_8hi` organisation at the 6.4 Gb/s
    /// JESD238 speed bin (the only HBM3 bin ramulator 2.1 ships): 32-bit
    /// channels with BL8, so one transfer moves 32 bytes.
    pub fn hbm3_preset(num_channels: usize) -> Result<Self> {
        Self::new(Self::hbm3_config(num_channels))
    }

    /// The configuration behind [`Ramulator::hbm3_preset`].
    pub fn hbm3_config(num_channels: usize) -> Config {
        let hbm3 = hbm3::HBM3 {
            timing: hbm3::HBM3Timing::HBM3_6400MBPS,
            org: hbm3::HBM3Org::HBM3_8GB_8HI,
        };

        let controller = Controller::HBM34(DDRController {
            options: Default::default(),
            scheduler: Scheduler::FrFcFs,
            refresh_manager: RefreshManager::all_bank(),
            row_policy: RowPolicy::Open,
            addr_mapper: AddrMapper::MOP4CLXOR,
            dram: DRAM::HBM3(hbm3.clone()),
        });

        Config {
            controllers: vec![controller; num_channels],
            channel_mapper: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use runtime::Duration;

    use super::*;

    #[test]
    fn hbm3_config_has_one_hbm34_controller_per_channel() {
        let config = Ramulator::hbm3_config(16);
        assert_eq!(config.controllers.len(), 16);
        assert!(config.controllers.iter().all(|controller| matches!(
            controller,
            Controller::HBM34(DDRController {
                dram: DRAM::HBM3(_),
                ..
            })
        )));
    }

    #[test]
    fn hbm3_preset_builds_with_sixteen_channels() {
        let ram = Ramulator::hbm3_preset(16).unwrap();
        assert_eq!(ram.num_channels(), 16);
        // Half-CK ticks at 6.4 Gb/s: 625 ps / 2.
        assert_eq!(ram.period(), Duration::from_picos(312));
        // BL8 on a 32-bit channel.
        assert_eq!(ram.transfer_size(), 32);
    }

    #[test]
    fn hbm2_preset_keeps_its_shape() {
        let ram = Ramulator::hbm2_preset(8).unwrap();
        assert_eq!(ram.num_channels(), 8);
        assert_eq!(ram.period(), Duration::from_picos(1000));
        assert_eq!(ram.transfer_size(), 16);
    }
}
