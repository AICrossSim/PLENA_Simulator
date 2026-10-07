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
        Self::hbm2_diagnostic(num_channels, hbm2::HbmDiagnostic::Native)
    }

    pub fn hbm2_diagnostic(num_channels: usize, diagnostic: hbm2::HbmDiagnostic) -> Result<Self> {
        let hbm2 = super::config::hbm2::HBM2 {
            timing: hbm2::HBM2Timing::HBM2_2000MBPS,
            org: hbm2::HBM2Org::HBM2_8GB,
            diagnostic,
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

#[cfg(test)]
mod diagnostic_probes {
    use super::*;
    use memory::MemoryTimingModel;
    use runtime::{Duration, Executor, Instant};
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

    async fn probe(profile: hbm2::HbmDiagnostic, requests: u64) -> u64 {
        let ram = Ramulator::hbm2_diagnostic(1, profile).unwrap()
            .with_issue_policy(crate::model::IssuePolicy::PerChannel, Duration::from_picos(1000));
        assert_eq!(ram.transfer_size(), 32);
        assert_eq!(ram.period().as_picos(), if profile == hbm2::HbmDiagnostic::ClockX2 { 500 } else { 1000 });
        let ex = Executor::new();
        let last = Arc::new(AtomicU64::new(0));
        for i in 0..requests {
            let r = ram.clone();
            let end = last.clone();
            ex.spawn(async move {
                r.read_mask(i / 2 * 64, if i % 2 == 0 { 1 } else { 2 }).await;
                end.fetch_max(Executor::current().now().as_picos(), Ordering::Relaxed);
            });
        }
        ex.enter(Instant::ETERNITY).await;
        assert_eq!(ram.telemetry()["native_pending"], 0);
        assert_eq!(ram.telemetry()["accepted_per_channel"], serde_json::json!([requests]));
        last.load(Ordering::Relaxed)
    }

    #[tokio::test]
    async fn native_diagnostic_profiles_change_service_without_changing_transactions() {
        use hbm2::HbmDiagnostic::*;
        let mut values = Vec::new();
        for profile in [Native, ColumnX2, ReturnLatencyX2, ColumnAndReturnX2, AllTimingX2, ClockX2] {
            let first = probe(profile, 1).await;
            let stream = probe(profile, 512).await;
            eprintln!("HBM_DIAGNOSTIC {profile:?} first_ps={first} stream512_ps={stream}");
            values.push((first, stream));
        }
        assert_eq!(values[0].0, values[1].0);
        assert_eq!(values[0].0 - values[2].0, 8_000);
        assert!(values[1].1 < values[0].1);
        assert!(values[4].0 < values[0].0);
    }
}
