mod accelerator;
mod cli;
mod dma;
mod load_config;
mod matrix_core;
mod matrix_machine;
mod op;
mod runner;
mod runtime_config;
mod stage_profile;
mod timing_overlay;
mod vector_machine;

use runtime::{Duration, Executor, Instant};

#[macro_export]
macro_rules! cycle {
    ($cycle: expr) => {
        ::runtime::Executor::current()
            .resolve_at($crate::runtime_config::PERIOD * ($cycle as u32))
            .await;
    };
}

#[tokio::main]
async fn main() {
    let executor = Executor::new();
    executor.spawn(runner::run_from_cli());
    executor.enter(Instant::ETERNITY).await;
    let latency = executor.now() - Instant::INIT;
    let cycles = latency
        .as_picos()
        .div_ceil(runtime_config::PERIOD.as_picos().max(1));
    if let Some(adjusted_cycles) = timing_overlay::experimental_report_cycles() {
        let adjusted_latency =
            Duration::from_picos(runtime_config::PERIOD.as_picos() * adjusted_cycles);
        // Report the serial (no-overlap) figure first, then the overlap-adjusted
        // one below. `latency` was computed above as `now - INIT`.
        tracing::info!(
            "Experimental overlap: serial latency {:?} cycles {}",
            latency,
            cycles
        );
        tracing::info!(
            "Simulation completed. Latency {:?} cycles {}",
            adjusted_latency,
            adjusted_cycles
        );
        return;
    }
    tracing::info!(
        "Simulation completed. Latency {:?} cycles {}",
        executor.now(),
        cycles
    );
}
