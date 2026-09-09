//! Numerical and finite-resource checks for decoupled output contexts.
//! The serialized 64-byte test bus is a deterministic fixture, not HBM evidence.
use super::tests::{fixture, reference, refined_architecture, simulate};
use super::*;

fn pool_architecture(contexts: usize, stages: usize) -> Architecture {
    let mut a = refined_architecture(2);
    for core in &mut a.cores {
        core.accumulator_bytes = 16_384;
        let r = core.refinement.as_mut().unwrap();
        r.operand_latch_bytes = stages * core.blen * core.mlen * 2;
        r.output_pool = Some(OutputPoolConfig {
            output_contexts: contexts,
            operand_stages: stages,
            scheduler_cycles: 1,
        });
    }
    a
}

#[test]
fn pool_rejects_an_incomplete_m_cohort_before_execution() {
    let (w, bytes, _) = fixture();
    let mut a = pool_architecture(4, 2);
    a.cores[0].refinement.as_mut().unwrap().m_rows = 1;
    validate(&w, &a, bytes.len() as u64).unwrap();
    a.cores[0]
        .refinement
        .as_mut()
        .unwrap()
        .output_pool
        .as_mut()
        .unwrap()
        .output_contexts = 3;
    // Expert 0 has four rows. Splitting it into uncharged weight rereads is
    // forbidden, even though a three-context tile is individually executable.
    assert!(validate(&w, &a, bytes.len() as u64).is_err());
}

#[test]
fn pool_scheduler_and_operand_stages_have_finite_positive_costs() {
    let (w, bytes, _) = fixture();
    let a = pool_architecture(8, 2);
    validate(&w, &a, bytes.len() as u64).unwrap();
    for (cycles, stages) in [(0, 2), (1, 0), (1, 3)] {
        let mut bad = a.clone();
        let p = bad.cores[0]
            .refinement
            .as_mut()
            .unwrap()
            .output_pool
            .as_mut()
            .unwrap();
        p.scheduler_cycles = cycles;
        p.operand_stages = stages;
        assert!(validate(&w, &bad, bytes.len() as u64).is_err());
    }
    let mut bad = a;
    let core = &mut bad.cores[0];
    // Exactly one stationary operand tile cannot fund two operand stages.
    core.refinement.as_mut().unwrap().operand_latch_bytes = core.blen * core.mlen * 2;
    assert!(validate(&w, &bad, bytes.len() as u64).is_err());
}

#[test]
fn pool_capacity_boundary_includes_control_pending_and_packed_scales() {
    let (w, bytes, _) = fixture();
    let mut a = pool_architecture(8, 2);
    let core = &mut a.cores[0];
    let r = core.refinement.as_ref().unwrap();
    let p = r.output_pool.as_ref().unwrap();
    let pipeline = core.blen * (core.mlen.ilog2() as usize + a.mac_pipeline_cycles as usize) * 4;
    let control = 128 * p.output_contexts + 64 * (core.weight_slots + p.operand_stages);
    let pending = p.output_contexts * r.m_rows * core.blen * 4;
    // The largest assigned expert has Me=4. Neither full FP32 output storage
    // nor the independent pending result records may be silently omitted.
    core.accumulator_bytes = 4 * 11 * 4 + pipeline + control + pending;
    core.weight_sram_bytes =
        core.weight_slots * core.blen * (core.mlen / 8) * 25 + r.operand_latch_bytes;
    validate(&w, &a, bytes.len() as u64).unwrap();
    let mut bad = a.clone();
    bad.cores[0].accumulator_bytes -= 1;
    assert!(validate(&w, &bad, bytes.len() as u64).is_err());
    let mut bad = a;
    bad.cores[0].weight_sram_bytes -= 1;
    assert!(validate(&w, &bad, bytes.len() as u64).is_err());
}

#[tokio::test]
async fn pool_nonsquare_m_n_k_tails_match_independent_bf16_reference() {
    let (mut w, bytes, dense) = fixture();
    w.shared_expert = Some(SharedExpert {
        expert: 3,
        weight: 0.5,
    });
    let expected = reference(&w, &dense);
    // D=9, F=11, P=3/2 and R=16/8 exercise both projection orientations,
    // alternating scales, nonzero padding, and M tails including shared Me=7.
    for contexts in [8, 32] {
        for stages in [1, 2] {
            for mt in [1, 3, 5] {
                let mut a = pool_architecture(contexts, stages);
                for core in &mut a.cores {
                    core.refinement.as_mut().unwrap().m_rows = mt;
                    core.accumulator_bytes = 16_384;
                }
                let report = simulate(w.clone(), a.clone(), &bytes).await;
                assert_eq!(
                    report.output_bf16, expected,
                    "Q={contexts} stages={stages} Mt={mt}"
                );
                assert_eq!(report.useful_macs, 3 * 14 * 9 * 11);
                for (observed, core) in report.cores.iter().zip(&a.cores) {
                    assert!(observed.weight_slots_peak <= core.weight_slots);
                    assert!(observed.weight_sram_peak_bytes <= core.weight_sram_bytes);
                    assert!(observed.accumulator_peak_bytes <= core.accumulator_bytes);
                }
            }
        }
    }
}

#[tokio::test]
async fn pool_reuses_each_weight_tile_across_the_complete_m_cohort() {
    let (mut w, bytes, dense) = fixture();
    let mut a = pool_architecture(8, 2);
    a.cores.truncate(1);
    a.small_core = 0;
    a.dispatch_threshold = 1;
    a.cores[0].refinement.as_mut().unwrap().m_rows = 1;
    w.routes.retain(|r| r.expert == 0);
    let four = simulate(w.clone(), a.clone(), &bytes).await;
    assert_eq!(four.output_bf16, reference(&w, &dense));
    w.routes.truncate(1);
    let one = simulate(w.clone(), a, &bytes).await;
    assert_eq!(one.output_bf16, reference(&w, &dense));
    assert_eq!(four.useful_macs, 4 * one.useful_macs);
    // Same expert, three projections, identical weight tiles; all four M rows
    // must consume each tile before its finite stage is released.
    assert_eq!(four.hbm_read_bytes, one.hbm_read_bytes);
}

#[tokio::test]
async fn pool_larger_output_ring_keeps_operand_and_weight_budgets_fixed() {
    let (w, bytes, dense) = fixture();
    let small = pool_architecture(8, 2);
    let large = pool_architecture(32, 2);
    let small_report = simulate(w.clone(), small.clone(), &bytes).await;
    let large_report = simulate(w.clone(), large.clone(), &bytes).await;
    assert_eq!(large_report.output_bf16, reference(&w, &dense));
    assert_eq!(small_report.hbm_read_bytes, large_report.hbm_read_bytes);
    for ((a, b), (ca, cb)) in small_report
        .cores
        .iter()
        .zip(&large_report.cores)
        .zip(small.cores.iter().zip(&large.cores))
    {
        assert_eq!(ca.weight_slots, cb.weight_slots);
        assert_eq!(ca.weight_sram_bytes, cb.weight_sram_bytes);
        assert_eq!(
            a.refinement.as_ref().unwrap().operand_latch_reserved_bytes,
            b.refinement.as_ref().unwrap().operand_latch_reserved_bytes
        );
        assert!(b.weight_slots_peak <= cb.weight_slots);
        assert!(b.accumulator_peak_bytes > a.accumulator_peak_bytes);
    }
}

#[tokio::test]
async fn pool_two_stages_drain_with_slow_feedback_and_one_accumulator_port() {
    let (w, bytes, dense) = fixture();
    let mut a = pool_architecture(8, 2);
    a.mac_pipeline_cycles = 100;
    a.global_dma_credits = 1;
    a.global_dma_staging_bytes = 64;
    for core in &mut a.cores {
        core.accumulator_bytes = 16_384;
        let r = core.refinement.as_mut().unwrap();
        r.m_rows = 1;
        r.accumulator_elements_per_cycle = 1;
        r.weight_read_elements_per_cycle = 1;
    }
    let report = simulate(w.clone(), a, &bytes).await;
    assert_eq!(report.output_bf16, reference(&w, &dense));
    assert_eq!(report.job_completions.len(), 3);
    assert!(
        report.total_ps < 100_000_000,
        "tiny finite workload must drain"
    );
    assert!(report.cores.iter().all(|c| c.weight_slots_peak <= 2));
    assert!(
        report
            .cores
            .iter()
            .any(|c| { c.accumulator_dependency_stall_ps + c.pipeline_drain_ps > 0 })
    );
}

#[tokio::test]
async fn pool_one_context_one_stage_one_slot_has_an_independent_writeback_path() {
    let (w, bytes, dense) = fixture();
    let mut a = pool_architecture(1, 1);
    a.mac_pipeline_cycles = 100;
    for core in &mut a.cores {
        core.weight_slots = 1;
        let r = core.refinement.as_mut().unwrap();
        r.m_rows = 4; // All assigned Me cohorts fit exactly one output record.
        r.accumulator_elements_per_cycle = 1;
        r.weight_read_elements_per_cycle = 1;
    }
    let report = simulate(w.clone(), a, &bytes).await;
    assert_eq!(report.output_bf16, reference(&w, &dense));
    assert_eq!(report.job_completions.len(), 3);
    for core in &report.cores {
        assert_eq!(core.weight_slots_peak, 1);
        let pool = core
            .refinement
            .as_ref()
            .unwrap()
            .output_pool
            .as_ref()
            .unwrap();
        assert_eq!(pool.contexts_peak, 1);
        assert_eq!(pool.operand_stages_peak, 1);
        assert!(pool.pending_contexts_peak <= 1);
    }
}

#[tokio::test]
async fn legacy_three_n_control_is_numerical_and_charges_three_latches() {
    let (w, bytes, dense) = fixture();
    let mut a = refined_architecture(3);
    for core in &mut a.cores {
        core.weight_slots = 3;
    }
    let report = simulate(w.clone(), a.clone(), &bytes).await;
    assert_eq!(report.output_bf16, reference(&w, &dense));
    for (observed, core) in report.cores.iter().zip(&a.cores) {
        assert_eq!(
            observed
                .refinement
                .as_ref()
                .unwrap()
                .operand_latch_reserved_bytes,
            3 * core.blen * core.mlen * 2
        );
        assert!(observed.weight_slots_peak <= 3);
    }
}

#[tokio::test]
async fn pool_projection_service_records_reconcile_without_adding_overlapping_waits() {
    let (w, bytes, _) = fixture();
    let a = pool_architecture(8, 2);
    let report = simulate(w, a.clone(), &bytes).await;
    for (core, config) in report.cores.iter().zip(&a.cores) {
        assert_eq!(core.projections.len(), 3 * core.jobs);
        assert_eq!(
            core.projections
                .iter()
                .map(|p| p.metrics.useful_macs)
                .sum::<u64>(),
            core.useful_macs
        );
        assert_eq!(
            core.projections
                .iter()
                .map(|p| p.metrics.compute_busy_ps)
                .sum::<u64>(),
            core.compute_busy_ps
        );
        assert_eq!(
            core.projections
                .iter()
                .map(|p| p.metrics.hbm_read_bytes)
                .sum::<u64>(),
            core.hbm_read_bytes
        );
        let mut last_end = 0;
        for projection in &core.projections {
            assert!(last_end <= projection.start_ps);
            last_end = projection.end_ps;
            let elapsed = projection.end_ps - projection.start_ps;
            let m = &projection.metrics;
            assert_eq!(
                m.tile_loads.count as usize,
                projection.n.div_ceil(config.blen) * projection.k.div_ceil(config.mlen)
            );
            assert_eq!(m.tile_admissions, m.tile_loads.count);
            assert_eq!(
                m.context_updates as usize,
                m.tile_loads.count as usize
                    * projection
                        .m
                        .div_ceil(config.refinement.as_ref().unwrap().m_rows)
            );
            assert!(m.compute_busy_ps <= elapsed);
            assert!(m.weight_port_busy_ps <= elapsed);
            assert!(m.accumulator_port_busy_ps <= elapsed);
            assert!(m.scheduler_busy_ps <= elapsed);
            assert_eq!(m.scheduler_busy_ps, m.scheduler_visits * a.clock_period_ps);
            assert!(m.tile_loads.min_ps <= m.tile_loads.max_ps);
            assert!(m.tile_loads.total_ps >= m.tile_loads.count * m.tile_loads.min_ps);
            assert!(m.tile_loads.total_ps <= m.tile_loads.count * m.tile_loads.max_ps);
        }
        assert!(last_end <= report.total_ps);
    }
}
