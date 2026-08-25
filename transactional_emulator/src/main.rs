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
mod timing;
mod vector_machine;

use runtime::{Executor, Instant};

// A simulated run is allocation-bound in the small-object range: every memory
// transfer takes a completion channel and a boxed callback, tens of millions of
// times. glibc malloc showed up at ~8% of a macro-benchmark profile.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[macro_export]
macro_rules! cycle {
    ($cycle: expr) => {
        $crate::timing::charge_cycles($cycle as u32).await;
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
