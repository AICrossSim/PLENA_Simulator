use core::sync::atomic::AtomicU32;

use anyhow::Result;
use ramulator::config::Config;
use ramulator::raw::Ramulator;

pub fn main() -> Result<()> {
    let clock = AtomicU32::new(0);
    let mut ramulator = Ramulator::new(Config::from_yaml(
        "
DRAM:
  impl: DDR4
  org:
    preset: DDR4_8Gb_x8
    channel: 2
  timing:
    preset: DDR4_2400P

Controller:
  impl: Generic
  Scheduler:
    impl: FRFCFS
  RefreshManager:
    impl: AllBank
  RowPolicy:
    impl: OpenRowPolicy
    cap: 4
  plugins:

AddrMapper:
  impl: RoBaRaCoCh
",
    )?)?;

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
