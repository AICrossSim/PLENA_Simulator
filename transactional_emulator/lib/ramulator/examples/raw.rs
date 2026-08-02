use core::sync::atomic::AtomicU32;

use anyhow::Result;
use ramulator::config::{
    AddrMapper, Config, Controller, DDRController, DRAM, RefreshManager, RowPolicy, Scheduler, ddr4,
};
use ramulator::raw::Ramulator;

pub fn main() -> Result<()> {
    let clock = AtomicU32::new(0);

    let ddr4 = ddr4::DDR4 {
        timing: ddr4::DDR4Timing::DDR4_2400P,
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

    let mut ramulator = Ramulator::new(Config {
        controllers: vec![controller],
        channel_mapper: Default::default(),
    })?;

    let freq = ramulator.period();
    println!("Clock period is {}ps", freq);

    for _ in 0..32 {
        ramulator.read(0, || {
            println!(
                "Callback {}!",
                clock.load(core::sync::atomic::Ordering::Relaxed)
            );
        });
    }

    let now = std::time::Instant::now();
    for _ in 0..1_000_000 {
        clock.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        ramulator.tick();
    }
    let t = now.elapsed().as_secs_f64();
    println!("elapsed {}s", t);

    Ok(())
}
