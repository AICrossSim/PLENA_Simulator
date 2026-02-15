//! Test utility only supposed for testing and benchmarking.
#![doc(hidden)]

use std::sync::Arc;
use std::time::Instant as WallInstant;

use rand::prelude::*;
use runtime::{Executor, Instant};
use tokio::sync::Semaphore;

use crate::MemoryTimingModel;

pub async fn sequential<T: MemoryTimingModel + 'static>(model: T, size: u64) {
    let model = Arc::new(model);
    let executor = Executor::new();

    executor.spawn(async move {
        let pacer = Arc::new(Semaphore::new(64));
        for offset in (0..size).step_by(64) {
            let permit = pacer.clone().acquire_owned().await.unwrap();
            let model_clone = model.clone();
            Executor::current().spawn(async move {
                model_clone.read(offset).await;
                drop(permit);
                // println!("{offset} {:?}", Executor::current().now());
            });
        }
    });

    let start = WallInstant::now();
    executor.enter(Instant::ETERNITY).await;
    let elapsed = start.elapsed();
    eprintln!(
        "Simulation completed. Last instance {:?}. Wall clock time {:?}",
        executor.now(),
        elapsed
    );
}

pub async fn sequential_1m<T: MemoryTimingModel + 'static>(model: T) {
    sequential(model, 1024 * 1024).await;
}

pub async fn sequential_1g<T: MemoryTimingModel + 'static>(model: T) {
    sequential(model, 1024 * 1024 * 1024).await;
}

pub async fn random<T: MemoryTimingModel + 'static>(model: T, size: u64) {
    let model = Arc::new(model);
    let executor = Executor::new();

    executor.spawn(async move {
        let pacer = Arc::new(Semaphore::new(64));
        // Use a fixed seed RNG for same sequence across runs.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in (0..size).step_by(64) {
            let permit = pacer.clone().acquire_owned().await.unwrap();
            let offset_remap = rng.random_range(0..size / 64) * 64;
            let model_clone = model.clone();
            Executor::current().spawn(async move {
                model_clone.read(offset_remap).await;
                drop(permit);
                // println!("{offset_remap} {:?}", Executor::current().now());
            });
        }
    });

    let start = WallInstant::now();
    executor.enter(Instant::ETERNITY).await;
    let elapsed = start.elapsed();
    eprintln!(
        "Simulation completed. Last instance {:?}. Wall clock time {:?}",
        executor.now(),
        elapsed
    );
}

pub async fn random_1m<T: MemoryTimingModel + 'static>(model: T) {
    random(model, 1024 * 1024).await
}

pub async fn random_1g<T: MemoryTimingModel + 'static>(model: T) {
    random(model, 1024 * 1024 * 1024).await
}
