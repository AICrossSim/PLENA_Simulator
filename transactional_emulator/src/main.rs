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

#[cfg(test)]
mod timing_golden_tests {
    const FIXTURE: &str = include_str!("../testbench/timing_goldens/golden_workloads.json");

    #[test]
    fn timing_golden_fixture_pins_required_workloads() {
        for needle in [
            "\"gpt_synthetic_small\"",
            "\"qwen_synthetic_small\"",
            "\"gpt_real_layer0_tok1\"",
            "\"qwen_real_decoder_block\"",
            "\"sim_latency_cycles\": 147758",
            "\"sim_latency_cycles\": 256559",
            "\"sim_latency_cycles\": 34354344",
            "\"sim_latency_cycles\": 65073075",
        ] {
            assert!(FIXTURE.contains(needle), "missing fixture entry: {needle}");
        }
    }
}
