use std::sync::Arc;

use half::bf16;
use memory::{MemoryBacked, MemoryTimingModel, WithStats, WithTiming};
use runtime::{Duration, Executor};

use super::*;

/// One serialized shared transfer resource, with real bytes behind it. Unlike
/// a request-count-only test, wrong addresses or missing reads change results.
struct TestTiming {
    bus: tokio::sync::Semaphore,
}
impl MemoryTimingModel for TestTiming {
    async fn read(&self, _: u64) {
        let _permit = self.bus.acquire().await.unwrap();
        Executor::current()
            .resolve_at(Duration::from_picos(3_000))
            .await;
    }
    async fn write(&self, _: u64) {
        let _permit = self.bus.acquire().await.unwrap();
        Executor::current()
            .resolve_at(Duration::from_picos(3_000))
            .await;
    }
}

#[derive(Clone)]
struct DenseExpert {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

fn append_matrix(
    bytes: &mut Vec<u8>,
    rows: usize,
    cols: usize,
    expert: usize,
    projection: usize,
) -> (MatrixRegion, Vec<f32>) {
    bytes.resize(bytes.len().div_ceil(64) * 64, 0xa5);
    let element_base = bytes.len() as u64;
    let stride = cols.div_ceil(8) * 8;
    // Padding deliberately contains nonzero values; masked tails must ignore it.
    bytes.resize(bytes.len() + rows * stride, 0x38);
    let mut values = vec![0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let byte = if row % cols == col {
                0x38
            } else if (row + col + expert + projection).is_multiple_of(7) {
                0xb0
            } else {
                0x00
            };
            bytes[element_base as usize + row * stride + col] = byte;
            // Alternate explicit shared scales by row. All values exact BF16.
            let scale = if row % 2 == 0 { 1.0 } else { 0.5 };
            values[row * cols + col] = scale
                * match byte {
                    0x38 => 1.0,
                    0xb0 => -0.5,
                    _ => 0.0,
                };
        }
    }
    bytes.resize(bytes.len().div_ceil(64) * 64, 0xa5);
    let scale_base = bytes.len() as u64;
    let scale_stride = cols.div_ceil(8);
    for row in 0..rows {
        bytes.extend(std::iter::repeat_n(
            if row % 2 == 0 { 127 } else { 126 },
            scale_stride,
        ));
    }
    bytes.resize(bytes.len().div_ceil(64) * 64, 0xa5);
    (
        MatrixRegion {
            rows,
            cols,
            element_base,
            scale_base,
            element_row_stride: stride as u64,
            scale_row_stride: scale_stride as u64,
        },
        values,
    )
}

fn fixture() -> (Workload, Vec<u8>, Vec<DenseExpert>) {
    let d = 9;
    let e = 11;
    let mut bytes = Vec::new();
    let mut experts = Vec::new();
    let mut dense = Vec::new();
    for expert in 0..4 {
        let (gate, g) = append_matrix(&mut bytes, e, d, expert, 0);
        let (up, u) = append_matrix(&mut bytes, e, d, expert, 1);
        let (down, out) = append_matrix(&mut bytes, d, e, expert, 2);
        experts.push(Expert {
            id: expert,
            gate,
            up,
            down,
        });
        dense.push(DenseExpert {
            gate: g,
            up: u,
            down: out,
        });
    }
    let input: Vec<Vec<u16>> = (0..7)
        .map(|t| {
            (0..d)
                .map(|k| bf16::from_f32((t as f32 - k as f32 + 2.0) / 8.0).to_bits())
                .collect()
        })
        .collect();
    let routes = (0..7)
        .map(|token| Route {
            token,
            slot: 0,
            expert: if token < 4 {
                0
            } else if token < 6 {
                1
            } else {
                2
            },
            weight: 0.75,
        })
        .collect();
    (
        Workload {
            schema_version: 1,
            name: "unit_nonzero_tails".into(),
            hbm_file: "unused.bin".into(),
            input_dim: d,
            expert_hidden_dim: e,
            inputs_bf16: input,
            routes,
            experts,
            shared_expert: None,
            metadata: None,
            grouped_routes: None,
        },
        bytes,
        dense,
    )
}

fn architecture() -> Architecture {
    Architecture {
        schema_version: 1,
        name: "dual_normal".into(),
        cores: vec![
            CoreConfig {
                id: "large".into(),
                blen: 4,
                mlen: 16,
                vector_sram_bytes: 4096,
                accumulator_bytes: 4096,
                weight_sram_bytes: 4096,
            },
            CoreConfig {
                id: "small".into(),
                blen: 2,
                mlen: 8,
                vector_sram_bytes: 4096,
                accumulator_bytes: 4096,
                weight_sram_bytes: 4096,
            },
        ],
        dispatch_threshold: 4,
        large_core: 0,
        small_core: 1,
        global_dma_credits: 2,
        global_dma_staging_bytes: 128,
        combine_sram_bytes: 16384,
        clock_period_ps: 1000,
        mac_pipeline_cycles: 2,
        vector_elements_per_cycle: 16,
    }
}

async fn simulate(w: Workload, a: Architecture, bytes: &[u8]) -> RunReport {
    let backing = MemoryBacked::with_capacity(bytes.len());
    backing.with_data(|dst| dst.copy_from_slice(bytes));
    let memory = Arc::new(WithStats::new(WithTiming::new(
        TestTiming {
            bus: tokio::sync::Semaphore::new(1),
        },
        backing,
    )));
    let report = run(w, a, memory.clone(), bytes.len() as u64).await.unwrap();
    assert_eq!(report.hbm_read_bytes, memory.statistics().total_bytes_read);
    assert_eq!(report.hbm_write_bytes, 0);
    report
}

fn gemm_reference(x: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|col| {
            let mut sum = 0f32;
            for kk in 0..k {
                sum += x[kk] * w[col * k + kk];
            }
            bf16::from_f32(sum).to_f32()
        })
        .collect()
}

fn expert_reference(x: &[f32], w: &DenseExpert, d: usize, e: usize) -> Vec<f32> {
    let gate = gemm_reference(x, &w.gate, e, d);
    let up = gemm_reference(x, &w.up, e, d);
    let z: Vec<f32> = gate
        .into_iter()
        .zip(up)
        .map(|(g, u)| bf16::from_f32((g / (1.0 + (-g).exp())) * u).to_f32())
        .collect();
    gemm_reference(&z, &w.down, d, e)
}

fn reference(w: &Workload, dense: &[DenseExpert]) -> Vec<Vec<u16>> {
    let mut routes = w.routes.clone();
    routes.sort_by_key(|r| (r.token, r.slot));
    (0..w.inputs_bf16.len())
        .map(|token| {
            let x: Vec<f32> = w.inputs_bf16[token]
                .iter()
                .map(|v| bf16::from_bits(*v).to_f32())
                .collect();
            let mut sums = vec![0f32; w.input_dim];
            for r in routes.iter().filter(|r| r.token == token) {
                let y = expert_reference(&x, &dense[r.expert], w.input_dim, w.expert_hidden_dim);
                for (s, v) in sums.iter_mut().zip(y) {
                    *s += r.weight * v;
                }
            }
            if let Some(r) = &w.shared_expert {
                let y = expert_reference(&x, &dense[r.expert], w.input_dim, w.expert_hidden_dim);
                for (s, v) in sums.iter_mut().zip(y) {
                    *s += r.weight * v;
                }
            }
            sums.into_iter()
                .map(|v| bf16::from_f32(v).to_bits())
                .collect()
        })
        .collect()
}

#[tokio::test]
async fn heterogeneous_nonzero_full_moe_and_tail_masks_match_scalar_reference() {
    let (w, bytes, dense) = fixture();
    let expected = reference(&w, &dense);
    let report = simulate(w.clone(), architecture(), &bytes).await;
    assert_eq!(report.output_bf16, expected);
    assert!(report.output_f32.iter().flatten().any(|v| *v != 0.0));
    assert_eq!(
        report.useful_macs,
        (w.routes.len() * 3 * w.input_dim * w.expert_hidden_dim) as u64
    );
    assert!(report.issued_macs > report.useful_macs);
    assert!(report.cores.iter().all(|r| r.jobs > 0));
    assert!(report.cores.iter().all(|r| r.weight_slots_peak == 2));
    assert!(
        report
            .cores
            .iter()
            .all(|r| r.accumulator_peak_bytes > 0 && r.vector_sram_peak_bytes > 0)
    );
    for c in &report.cores {
        assert_eq!(c.compute_busy_ps, c.issued_macs / c.multipliers * 1000);
        assert!(c.mac_utilization > 0.0 && c.mac_utilization <= 1.0);
    }
}

#[tokio::test]
async fn baseline_has_same_numerical_path_and_multiplies_and_streaming_service() {
    let (w, bytes, _) = fixture();
    let dual = architecture();
    let mut single = dual.clone();
    single.name = "single".into();
    // 4*16 + 2*8 = 80 = 2*40, with a non-power-of-two MLEN.
    single.cores.truncate(1);
    single.cores[0].blen = 2;
    single.cores[0].mlen = 40;
    single.large_core = 0;
    single.small_core = 0;
    let a = simulate(w.clone(), dual, &bytes).await;
    let b = simulate(w, single, &bytes).await;
    assert_eq!(a.multipliers, b.multipliers);
    assert_eq!(a.output_bf16, b.output_bf16);
    assert_eq!(a.useful_macs, b.useful_macs);
    assert_eq!(b.cores[0].weight_slots_peak, 2);
    assert_eq!(
        b.cores[0].compute_busy_ps,
        b.issued_macs / b.multipliers * 1000
    );
    assert!(b.cores[0].pipeline_drain_ps > 0);
}

#[tokio::test]
async fn duplicate_expert_routes_shared_expert_and_input_order_are_deterministic() {
    let (mut w, bytes, dense) = fixture();
    w.routes.push(Route {
        token: 0,
        slot: 3,
        expert: 0,
        weight: -0.25,
    });
    w.routes.push(Route {
        token: 0,
        slot: 1,
        expert: 2,
        weight: 0.5,
    });
    w.shared_expert = Some(SharedExpert {
        expert: 3,
        weight: 0.125,
    });
    let expected = reference(&w, &dense);
    let first = simulate(w.clone(), architecture(), &bytes).await;
    w.routes.reverse();
    let reversed = simulate(w, architecture(), &bytes).await;
    assert_eq!(first.output_bf16, expected);
    assert_eq!(first.output_bf16, reversed.output_bf16);
    assert_eq!(first.total_ps, reversed.total_ps);
    assert_eq!(first.job_completions.len(), 4);
    assert!(first.job_completions.iter().any(|j| j.shared));
}

#[tokio::test]
async fn global_credit_limit_is_shared_and_hbm_byte_counters_are_real() {
    let (w, bytes, _) = fixture();
    for credits in [1, 2, 3] {
        let mut a = architecture();
        a.global_dma_credits = credits;
        a.global_dma_staging_bytes = 64 * credits;
        let report = simulate(w.clone(), a, &bytes).await;
        assert_eq!(report.global_dma_inflight_peak, credits);
        assert_eq!(report.global_dma_staging_peak_bytes, credits * 64);
        assert_eq!(
            report.hbm_read_bytes,
            report.cores.iter().map(|r| r.hbm_read_bytes).sum::<u64>()
        );
        assert!(report.hbm_read_bytes > 0);
    }
}

#[test]
fn finite_capacity_shortages_fail_before_execution() {
    let (w, bytes, _) = fixture();
    let valid = architecture();
    validate(&w, &valid, bytes.len() as u64).unwrap();
    for resource in 0..5 {
        let mut a = valid.clone();
        match resource {
            0 => a.cores[0].vector_sram_bytes = 1,
            1 => a.cores[0].accumulator_bytes = 1,
            2 => a.cores[0].weight_sram_bytes = 1,
            3 => a.combine_sram_bytes = 1,
            4 => a.global_dma_staging_bytes = 1,
            _ => unreachable!(),
        }
        assert!(
            validate(&w, &a, bytes.len() as u64).is_err(),
            "resource {resource}"
        );
    }
}

#[test]
fn malformed_routes_geometry_and_hbm_ranges_are_rejected() {
    let (w, bytes, _) = fixture();
    let a = architecture();
    let mut bad = w.clone();
    bad.routes.push(bad.routes[0].clone());
    assert!(
        validate(&bad, &a, bytes.len() as u64)
            .unwrap_err()
            .contains("duplicate route slot")
    );
    let mut bad = w.clone();
    bad.experts[0].gate.scale_base = bytes.len() as u64;
    assert!(
        validate(&bad, &a, bytes.len() as u64)
            .unwrap_err()
            .contains("beyond HBM")
    );
    let mut bad = w.clone();
    bad.experts[0].gate.element_row_stride = 9;
    assert!(
        validate(&bad, &a, bytes.len() as u64)
            .unwrap_err()
            .contains("strides")
    );
    let mut bad = w.clone();
    bad.inputs_bf16[0][0] = bf16::NAN.to_bits();
    assert!(validate(&bad, &a, bytes.len() as u64).is_err());
    let mut bad_a = a;
    bad_a.cores[0].mlen = 7;
    assert!(validate(&w, &bad_a, bytes.len() as u64).is_err());
}

#[tokio::test]
async fn unselected_experts_do_not_read_hbm_and_empty_routes_return_zero() {
    let (mut w, bytes, _) = fixture();
    w.routes.clear();
    let report = simulate(w, architecture(), &bytes).await;
    assert_eq!(report.hbm_read_bytes, 0);
    assert_eq!(report.useful_macs, 0);
    assert!(report.output_bf16.iter().flatten().all(|v| *v == 0));
    assert!(report.job_completions.is_empty());
    assert!(report.shared_vector_busy_ps > 0);
}

#[tokio::test]
async fn out_of_order_completion_preserves_route_reduction_order() {
    let (mut w, bytes, dense) = fixture();
    // Large expert is assigned first, but receives many extra rows. Smaller
    // expert must complete while it is still running; combine cannot reorder.
    for slot in 1..100 {
        w.routes.push(Route {
            token: 0,
            slot,
            expert: 0,
            weight: 0.0625,
        });
    }
    let expected = reference(&w, &dense);
    let mut a = architecture();
    a.cores[0].vector_sram_bytes = 16384;
    a.cores[0].accumulator_bytes = 16384;
    let report = simulate(w, a, &bytes).await;
    assert_ne!(report.job_completions[0].job, 0);
    assert_eq!(report.output_bf16, expected);
    let large = report.job_completions.iter().find(|j| j.job == 0).unwrap();
    assert!(report.job_completions[0].output_copied_ps < large.output_copied_ps);
}

#[test]
fn serde_contract_rejects_misspelled_fields() {
    let (w, _, _) = fixture();
    let mut value = serde_json::to_value(w).unwrap();
    value["hidden_dimm"] = serde_json::json!(9);
    assert!(serde_json::from_value::<Workload>(value).is_err());
}

#[test]
fn extreme_clock_and_invalid_square_mapping_fail_before_timers() {
    let (w, bytes, _) = fixture();
    let mut a = architecture();
    a.cores.truncate(1);
    a.large_core = 0;
    a.small_core = 0;
    a.cores[0].blen = 1;
    a.cores[0].mlen = 8;
    a.mac_pipeline_cycles = 0;
    a.clock_period_ps = u64::MAX / 4;
    a.vector_elements_per_cycle = 1;
    assert!(
        validate(&w, &a, bytes.len() as u64)
            .unwrap_err()
            .contains("duration exceeds")
    );
    a.clock_period_ps = 1000;
    a.cores[0].blen = 3;
    assert!(
        validate(&w, &a, bytes.len() as u64)
            .unwrap_err()
            .contains("divisible by BLEN")
    );
}
