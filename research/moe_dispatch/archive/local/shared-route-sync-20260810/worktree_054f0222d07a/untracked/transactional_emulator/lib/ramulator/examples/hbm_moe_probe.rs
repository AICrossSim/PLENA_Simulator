//! Scratch probe: what actually limits PLENA's achieved HBM read bandwidth?
//!
//! Three questions:
//!   A) how does 8-channel HBM2 read bandwidth scale with outstanding requests?
//!   B) does splitting one stream into N concurrent streams help, at fixed outstanding?
//!   C) how much does PLENA's split element/scale address pattern cost vs inlined scales?

use std::sync::Arc;

use memory::MemoryTimingModel;
use ramulator::Ramulator;
use runtime::{Executor, Instant};
use tokio::sync::Semaphore;

const SIZE: u64 = 16 * 1024 * 1024; // 16 MiB per measurement

/// Issue `addrs` as 64B reads with at most `outstanding` in flight; return GB/s.
async fn measure(model: Arc<Ramulator>, addrs: Vec<u64>, outstanding: usize) -> (f64, f64) {
    let executor = Executor::new();
    let bytes = (addrs.len() as u64) * 64;
    let result = Arc::new(std::sync::Mutex::new(0.0f64));
    let r = result.clone();

    executor.spawn(async move {
        let pacer = Arc::new(Semaphore::new(outstanding));
        let start = Executor::current().now();
        for a in addrs {
            let permit = pacer.clone().acquire_owned().await.unwrap();
            let m = model.clone();
            Executor::current().spawn(async move {
                m.read(a).await;
                drop(permit);
            });
        }
        let _all = pacer.acquire_many(outstanding as u32).await.unwrap();
        let secs = (Executor::current().now() - start).as_picos() as f64 / 1e12;
        *r.lock().unwrap() = secs;
    });
    executor.enter(Instant::ETERNITY).await;

    let secs = *result.lock().unwrap();
    let gbps = (bytes as f64 / secs) / 1e9;
    (gbps, gbps / 128.0 * 100.0) // 8-channel HBM2 peak = 128 GB/s
}

fn seq(n: u64) -> Vec<u64> {
    (0..n).map(|i| i * 64).collect()
}

/// `streams` interleaved sequential regions, 1 GiB apart (different experts).
fn multi(n: u64, streams: u64) -> Vec<u64> {
    let per = n / streams;
    let mut v = Vec::with_capacity(n as usize);
    for i in 0..per {
        for s in 0..streams {
            v.push(s * (1 << 30) + i * 64);
        }
    }
    v
}

/// PLENA today: per MLEN=64 row, one 64B element burst, plus one 64B scale burst
/// from a region `elem_region_len` away.  Scale rows are `row_stride/8` apart, so
/// consecutive rows' scales land in different 64B lines.
fn plena_split(rows: u64, row_stride: u64, elem_region_len: u64) -> Vec<u64> {
    let mut v = Vec::with_capacity((rows * 2) as usize);
    for r in 0..rows {
        v.push(r * row_stride); // element burst
        v.push(elem_region_len + (r * row_stride) / 8); // scale burst, far away
    }
    v
}

/// Proposed: scales inlined right after each element block, one contiguous stream.
fn plena_inlined(rows: u64) -> Vec<u64> {
    // 8 element bursts (512B) then 1 scale burst (64B) covering them.
    let mut v = Vec::new();
    let mut a = 0u64;
    for _ in 0..(rows / 8) {
        for _ in 0..8 {
            v.push(a);
            a += 64;
        }
        v.push(a); // the shared scale line
        a += 64;
    }
    v
}

/// N 条流，但每条流连续发 `chunk` 个 64B 请求才切换（保住 DRAM row locality）
fn multi_chunked(n: u64, streams: u64, chunk: u64) -> Vec<u64> {
    let per = n / streams;
    let mut v = Vec::with_capacity(n as usize);
    let mut base = 0u64;
    while base < per {
        for s in 0..streams {
            for c in 0..chunk.min(per - base) {
                v.push(s * (1 << 30) + (base + c) * 64);
            }
        }
        base += chunk;
    }
    v
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let n = SIZE / 64;
    println!("8-channel HBM2, peak = 128 GB/s, {} MiB per point\n", SIZE >> 20);

    println!("=== A0) 访问模式：连续 vs PLENA 现在的跳读，outstanding=256 ===");
    println!("{:>34} {:>10} {:>9}", "模式", "GB/s", "峰值%");
    {
        let m = Arc::new(Ramulator::hbm2_preset(8)?);
        let (g, p) = measure(m, seq(n), 256).await;
        println!("{:>34} {:>10.2} {:>8.1}%", "连续（理想 tiled 布局）", g, p);
    }
    for &stride in &[1408u64, 2048, 512] {
        // 一个 tile = 64 行 x 64B，行间隔 = 矩阵一行的字节数
        let mut v = Vec::with_capacity(n as usize);
        let tiles = n / 64;
        for t in 0..tiles {
            let tile_base = (t / 22) * 64 * stride + (t % 22) * 64;   // (k_blk, n_blk)
            for r in 0..64u64 {
                v.push(tile_base + r * stride);
            }
        }
        let m = Arc::new(Ramulator::hbm2_preset(8)?);
        let (g, p) = measure(m, v, 256).await;
        println!("{:>34} {:>10.2} {:>8.1}%", format!("行主序，stride={}B", stride), g, p);
    }

    println!("\n=== A) 顺序读 vs outstanding 请求数 ===");
    println!("{:>12} {:>12} {:>10}", "outstanding", "GB/s", "峰值%");
    for &o in &[1usize, 4, 16, 64, 128, 256, 512, 1024, 2048] {
        let m = Arc::new(Ramulator::hbm2_preset(8)?);
        let (g, p) = measure(m, seq(n), o).await;
        println!("{:>12} {:>12.2} {:>9.1}%", o, g, p);
    }

    println!("\n=== B) 多流（模拟并发专家），固定 outstanding=256 ===");
    println!("{:>12} {:>12} {:>10}", "流数", "GB/s", "峰值%");
    for &s in &[1u64, 2, 4, 8, 16] {
        let m = Arc::new(Ramulator::hbm2_preset(8)?);
        let (g, p) = measure(m, multi(n, s), 256).await;
        println!("{:>12} {:>12.2} {:>9.1}%", s, g, p);
    }

    println!("\n=== B2) 多流 + 粗粒度交错（每条流连发 chunk 个请求再切换），outstanding=256 ===");
    println!("{:>8} {:>10} {:>12} {:>10}", "流数", "chunk", "GB/s", "峰值%");
    for &s in &[4u64, 8] {
        for &c in &[1u64, 4, 16, 64, 256] {
            let m = Arc::new(Ramulator::hbm2_preset(8)?);
            let (g, p) = measure(m, multi_chunked(n, s, c), 256).await;
            println!("{:>8} {:>10} {:>12.2} {:>9.1}%", s, format!("{}B", c*64), g, p);
        }
    }

    println!("\n=== C) PLENA 的 element/scale 分离 vs 内联，outstanding=256 ===");
    let rows = n / 2; // split pattern emits 2 reads per row
    let m = Arc::new(Ramulator::hbm2_preset(8)?);
    let (g1, p1) = measure(m, plena_split(rows, 1408, 2048 * 1408), 256).await;
    println!("{:>28} {:>10.2} GB/s {:>8.1}%", "现在（scale 另一个区域）", g1, p1);
    let m = Arc::new(Ramulator::hbm2_preset(8)?);
    let (g2, p2) = measure(m, plena_inlined(n), 256).await;
    println!("{:>28} {:>10.2} GB/s {:>8.1}%", "内联（scale 紧跟元素）", g2, p2);
    println!("{:>28} {:>10.2}x  （只算带宽，不含少搬的 44% 字节）", "→ 带宽提升", g2 / g1);

    Ok(())
}
