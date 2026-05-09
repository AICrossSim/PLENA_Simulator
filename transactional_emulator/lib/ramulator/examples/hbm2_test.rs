use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::time::Instant as WallInstant;

use runtime::{Executor, Instant};
use tokio::sync::Semaphore;

use memory::MemoryTimingModel;

/// Test single read latency by issuing one read and measuring simulated time.
async fn test_single_read_latency<T: MemoryTimingModel + 'static>(model: T, name: &str) {
    let model = Arc::new(model);
    let executor = Executor::new();
    let name = name.to_string();

    let model_clone = model.clone();
    executor.spawn(async move {
        let start = Executor::current().now();
        model_clone.read(0).await;
        let end = Executor::current().now();
        let latency = end - start;
        println!("[{}] Single read latency: {:?}", name, latency);
    });

    executor.enter(Instant::ETERNITY).await;
}

/// Test sequential read bandwidth by reading a contiguous block of memory.
async fn test_sequential_bandwidth<T: MemoryTimingModel + 'static>(
    model: T,
    size: u64,
    name: &str,
) {
    let model = Arc::new(model);
    let executor = Executor::new();
    let name = name.to_string();

    let model_clone = model.clone();
    let name_clone = name.clone();
    executor.spawn(async move {
        // Use a semaphore to limit outstanding requests (simulating realistic memory controller queue)
        let pacer = Arc::new(Semaphore::new(64));

        let start = Executor::current().now();

        for offset in (0..size).step_by(64) {
            let permit = pacer.clone().acquire_owned().await.unwrap();
            let model_inner = model_clone.clone();
            Executor::current().spawn(async move {
                model_inner.read(offset).await;
                drop(permit);
            });
        }

        // Wait for all reads to complete by acquiring all permits
        let _all_permits = pacer.acquire_many(64).await.unwrap();

        let end = Executor::current().now();
        let elapsed_secs = (end - start).to_secs();
        let bandwidth_gbps = (size as f64 / elapsed_secs) / (1024.0 * 1024.0 * 1024.0);

        println!(
            "[{}] Sequential read: {} bytes in {:?} ({:.2} GB/s)",
            name_clone,
            size,
            end - start,
            bandwidth_gbps
        );
    });

    let wall_start = WallInstant::now();
    executor.enter(Instant::ETERNITY).await;
    let wall_elapsed = wall_start.elapsed();

    println!("[{}] Simulation wall clock time: {:?}", name, wall_elapsed);
}

/// Test random read bandwidth by reading random addresses.
async fn test_random_bandwidth<T: MemoryTimingModel + 'static>(model: T, size: u64, name: &str) {
    use rand::prelude::*;

    let model = Arc::new(model);
    let executor = Executor::new();
    let name = name.to_string();

    let model_clone = model.clone();
    let name_clone = name.clone();
    executor.spawn(async move {
        let pacer = Arc::new(Semaphore::new(64));
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let start = Executor::current().now();

        for _ in (0..size).step_by(64) {
            let permit = pacer.clone().acquire_owned().await.unwrap();
            let offset = rng.random_range(0..size / 64) * 64;
            let model_inner = model_clone.clone();
            Executor::current().spawn(async move {
                model_inner.read(offset).await;
                drop(permit);
            });
        }

        // Wait for all reads to complete
        let _all_permits = pacer.acquire_many(64).await.unwrap();

        let end = Executor::current().now();
        let elapsed_secs = (end - start).to_secs();
        let bandwidth_gbps = (size as f64 / elapsed_secs) / (1024.0 * 1024.0 * 1024.0);

        println!(
            "[{}] Random read: {} bytes in {:?} ({:.2} GB/s)",
            name_clone,
            size,
            end - start,
            bandwidth_gbps
        );
    });

    let wall_start = WallInstant::now();
    executor.enter(Instant::ETERNITY).await;
    let wall_elapsed = wall_start.elapsed();

    println!("[{}] Simulation wall clock time: {:?}", name, wall_elapsed);
}

/// Test write bandwidth
async fn test_write_bandwidth<T: MemoryTimingModel + 'static>(model: T, size: u64, name: &str) {
    let model = Arc::new(model);
    let executor = Executor::new();
    let name = name.to_string();

    let model_clone = model.clone();
    executor.spawn(async move {
        let pacer = Arc::new(Semaphore::new(64));

        let start = Executor::current().now();

        for offset in (0..size).step_by(64) {
            let permit = pacer.clone().acquire_owned().await.unwrap();
            let model_inner = model_clone.clone();
            Executor::current().spawn(async move {
                model_inner.write(offset).await;
                drop(permit);
            });
        }

        let _all_permits = pacer.acquire_many(64).await.unwrap();

        let end = Executor::current().now();
        let elapsed_secs = (end - start).to_secs();
        let bandwidth_gbps = (size as f64 / elapsed_secs) / (1024.0 * 1024.0 * 1024.0);

        println!(
            "[{}] Sequential write: {} bytes in {:?} ({:.2} GB/s)",
            name,
            size,
            end - start,
            bandwidth_gbps
        );
    });

    executor.enter(Instant::ETERNITY).await;
}

// Helper trait to get seconds from Duration
trait DurationExt {
    fn to_secs(&self) -> f64;
}

impl DurationExt for runtime::Duration {
    fn to_secs(&self) -> f64 {
        // Duration is stored in picoseconds internally
        // We need to convert through Instant to get the value
        let instant = runtime::Instant::INIT + *self;
        instant.to_secs()
    }
}

#[tokio::main]
async fn main() {
    println!("=== HBM2 Bandwidth and Latency Test ===\n");

    // HBM2 stack configurations:
    // - 1 stack = 8 channels
    // - 2 stacks = 16 channels
    // - 4 stacks = 32 channels
    let stack_configs = [
        (8, 1, "HBM2 1-stack (8 channels)"),
        (16, 2, "HBM2 2-stack (16 channels)"),
        (32, 4, "HBM2 4-stack (32 channels)"),
    ];

    // Data sizes
    let size_1mb = 1024 * 1024;
    let size_64mb = 64 * 1024 * 1024;
    let size_256mb = 256 * 1024 * 1024;

    println!("=== HBM2 Multi-Stack Performance Test ===\n");
    println!("HBM2 spec: 2 Gbps data rate, 64-bit channel width, 8 channels per stack\n");
    println!("Theoretical peak bandwidth per stack: 2 Gbps * 64 bits * 8 channels / 8 = 128 GB/s");
    println!("  - 1 stack:  128 GB/s theoretical");
    println!("  - 2 stacks: 256 GB/s theoretical");
    println!("  - 4 stacks: 512 GB/s theoretical");
    println!("\n");

    for (num_channels, num_stacks, name) in stack_configs {
        println!("\n{}", "=".repeat(60));
        println!("--- {} ---", name);
        println!("{}", "=".repeat(60));

        // Create HBM2 model
        let new_hbm2 =
            || ManuallyDrop::new(ramulator::Ramulator::hbm2_preset(num_channels).unwrap());

        // Print configuration
        {
            let model = new_hbm2();
            println!(
                "Configuration: {} stacks, {} channels, transfer_size={} bytes, period={:?}",
                num_stacks,
                num_channels,
                model.transfer_size(),
                model.period()
            );
            let theoretical_bw = 2.0 * 64.0 * num_channels as f64 / 8.0; // GB/s
            println!("Theoretical peak bandwidth: {:.0} GB/s", theoretical_bw);
        }

        println!("\n-- Latency Tests --");
        // Test single read latency
        test_single_read_latency(new_hbm2(), name).await;

        println!("\n-- Bandwidth Tests (1MB) --");
        // Test sequential bandwidth (1MB)
        test_sequential_bandwidth(new_hbm2(), size_1mb, name).await;

        // Test random bandwidth (1MB)
        test_random_bandwidth(new_hbm2(), size_1mb, name).await;

        // Test write bandwidth (1MB)
        test_write_bandwidth(new_hbm2(), size_1mb, name).await;

        println!("\n-- Bandwidth Tests (64MB) --");
        test_sequential_bandwidth(new_hbm2(), size_64mb, name).await;
        test_random_bandwidth(new_hbm2(), size_64mb, name).await;

        println!("\n-- Bandwidth Tests (256MB) --");
        test_sequential_bandwidth(new_hbm2(), size_256mb, name).await;
        test_random_bandwidth(new_hbm2(), size_256mb, name).await;
    }

    println!("\n\n{}", "=".repeat(60));
    println!("=== Comparison with DDR4 ===");
    println!("{}\n", "=".repeat(60));

    // Compare with DDR4
    let new_ddr4_1c = || ManuallyDrop::new(ramulator::Ramulator::ddr4_preset(1).unwrap());
    let new_ddr4_2c = || ManuallyDrop::new(ramulator::Ramulator::ddr4_preset(2).unwrap());

    {
        let model = new_ddr4_1c();
        println!(
            "DDR4 1-channel: transfer_size={} bytes, period={:?}",
            model.transfer_size(),
            model.period()
        );
    }

    test_single_read_latency(new_ddr4_1c(), "DDR4 1-channel").await;
    test_sequential_bandwidth(new_ddr4_1c(), size_1mb, "DDR4 1-channel").await;
    test_sequential_bandwidth(new_ddr4_2c(), size_1mb, "DDR4 2-channel").await;

    println!("\n\n{}", "=".repeat(60));
    println!("=== Summary ===");
    println!("{}", "=".repeat(60));
    println!("\nHBM2 characteristics:");
    println!("  - Data rate: 2 Gbps per pin");
    println!("  - Channel width: 64 bits (8 bytes)");
    println!("  - Burst size: 2 (internal prefetch)");
    println!("  - Transfer size per burst: 2 * 8 = 16 bytes");
    println!("  - Channels per stack: 8");
    println!("\nTheoretical peak bandwidth:");
    println!("  - 1 stack (8 ch):  128 GB/s");
    println!("  - 2 stacks (16 ch): 256 GB/s");
    println!("  - 4 stacks (32 ch): 512 GB/s");
    println!("\nNote: Actual bandwidth depends on access patterns, bank conflicts,");
    println!("      row buffer hits/misses, and memory controller scheduling.");
}
