mod accelerator;
mod cli;
mod dma;
mod load_config;
mod matrix_machine;
mod op;
mod opcode_timing;
mod profiler;
mod runner;
mod runtime_config;
mod scheduler;
mod timing;
mod vector_machine;

use std::sync::{Arc, Mutex};

use runtime::{Executor, Instant};

#[macro_export]
macro_rules! cycle {
    ($cycle: expr) => {
        ::runtime::Executor::current()
            .resolve_at(*$crate::runtime_config::PERIOD * ($cycle as u32))
            .await;
    };
}

#[tokio::main]
async fn main() {
    let executor = Executor::new();
    let reported_outcome: Arc<Mutex<Option<runner::RunOutcome>>> = Arc::new(Mutex::new(None));
    let reported_outcome_task = reported_outcome.clone();
    executor.spawn(async move {
        let outcome = runner::run_from_cli().await;
        *reported_outcome_task.lock().unwrap() = Some(outcome);
    });
    executor.enter(Instant::ETERNITY).await;
    let outcome = reported_outcome
        .lock()
        .unwrap()
        .unwrap_or_else(|| runner::RunOutcome {
            latency: executor.now() - Instant::INIT,
            rtl_validation_failed: false,
        });
    tracing::info!("Simulation completed. Latency {:?}", outcome.latency);
    if outcome.rtl_validation_failed {
        eprintln!(
            "error: rtl-v1 run contains unsupported or out-of-domain opcode timing; artifacts were written"
        );
        std::process::exit(2);
    }
}
