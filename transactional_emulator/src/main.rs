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
mod vector_machine;

use runtime::{Executor, Instant};

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
    tracing::info!(
        "Simulation completed. Latency {:?} cycles {}",
        executor.now(),
        cycles
    );
}
