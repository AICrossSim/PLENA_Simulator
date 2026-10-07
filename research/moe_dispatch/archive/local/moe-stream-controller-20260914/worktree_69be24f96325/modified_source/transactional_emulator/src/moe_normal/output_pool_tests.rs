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

fn event_architecture(contexts: usize, stages: usize, selection: ReadySelection) -> Architecture {
    let mut a = pool_architecture(contexts, stages);
    for c in &mut a.cores {
        c.refinement.as_mut().unwrap().stream_ctrl = Some(StreamControlConfig {
            event_ready: true,
            selection,
            ..Default::default()
        });
    }
    a
}

#[tokio::test]
async fn cohort_bursts_preserve_tail_k_order_bytes_and_reconcile_every_control_visit() {
    for (me, mt, feedback, ports, selection) in [
        (1, 1, 0, 1, ReadySelection::Rotating),
        (8, 1, 60, 1, ReadySelection::Rotating),
        (17, 1, 0, 1, ReadySelection::Lowest),
        (32, 1, 60, 2, ReadySelection::Rotating),
        (17, 4, 0, 2, ReadySelection::Lowest),
        (32, 4, 0, 1, ReadySelection::Rotating),
    ] {
        let (mut w, bytes, dense) = fixture();
        w.inputs_bf16 = vec![w.inputs_bf16[0].clone(); me];
        w.routes = (0..me)
            .map(|token| Route {
                token,
                slot: 0,
                expert: 0,
                weight: 1.0,
            })
            .collect();
        let mut a = event_architecture(32, 2, selection);
        a.dispatch_threshold = 1;
        a.mac_pipeline_cycles = feedback;
        for c in &mut a.cores {
            c.mlen = 8; // D=9/F=11 force ascending K segments and padded tails.
            c.weight_slots = 3;
            let r = c.refinement.as_mut().unwrap();
            r.m_rows = mt;
            r.operand_latch_bytes = 2 * c.blen * c.mlen * 2;
        }
        let old = simulate(w.clone(), a.clone(), &bytes).await;
        a.diagnostic.control_ports = ports;
        for c in &mut a.cores {
            c.refinement
                .as_mut()
                .unwrap()
                .stream_ctrl
                .as_mut()
                .unwrap()
                .cohort_control = true;
        }
        let one = simulate(w.clone(), a.clone(), &bytes).await;
        let two = simulate(w.clone(), a.clone(), &bytes).await;
        assert_eq!(
            serde_json::to_value(&one).unwrap(),
            serde_json::to_value(&two).unwrap()
        );
        assert_eq!(one.output_bf16, reference(&w, &dense));
        assert_eq!(one.hbm_read_bytes, old.hbm_read_bytes);
        for (got, expected) in [&one.output_f32, &one.pre_round_output_f32]
            .into_iter()
            .zip([&old.output_f32, &old.pre_round_output_f32])
        {
            assert_eq!(
                got.iter()
                    .flatten()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .flatten()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let c = &one.cores[0];
        let d = c.refinement.as_ref().unwrap();
        let s = d.stream_ctrl.as_ref().unwrap();
        let p = d.output_pool.as_ref().unwrap();
        let admission: u64 = c
            .projections
            .iter()
            .map(|x| x.metrics.band_admissions * (x.m.div_ceil(mt) as u64 + 1))
            .sum();
        assert_eq!(
            p.scheduler_visits,
            admission
                + 8 * p.tile_admissions
                + p.context_updates
                + 4 * s.burst_starts
                + s.completion_mask_cycles
        );
        assert_eq!(s.burst_starts, s.burst_completions);
        assert_eq!(
            s.event_counts_by_kind,
            [p.tile_admissions, p.tile_admissions, s.burst_starts]
        );
        assert_eq!(
            s.event_update_cycles_by_kind,
            [
                p.tile_admissions,
                p.context_updates,
                2 * s.burst_starts + s.completion_mask_cycles
            ]
        );
        assert_eq!(s.completed_contexts, p.context_updates);
        assert_eq!(s.writeback_order_checks, p.context_updates);
        assert!(s.sequencer_cycles >= p.context_updates);
        assert_eq!(s.sequencer_busy_ps, s.sequencer_cycles * a.clock_period_ps);
        assert_eq!(
            s.sequencer_cycles - p.context_updates,
            s.burst_not_ready_checks
        );
        assert!(s.completion_collectors_peak <= 32);
        assert_eq!(p.scheduler_busy_ps, p.scheduler_visits * a.clock_period_ps);
        assert!(
            s.control_occupancy_ps <= p.scheduler_busy_ps
                && p.scheduler_busy_ps <= s.control_occupancy_ps * ports as u64
        );
        assert_eq!(
            c.accumulator_peak_bytes,
            old.cores[0].accumulator_peak_bytes + (ports - 1) * 64
        );
    }
}

#[tokio::test]
async fn charged_n3_uses_identical_ports_and_exact_issue_install_completion_charges() {
    let (w, bytes, _) = fixture();
    let mut a = refined_architecture(3);
    for c in &mut a.cores {
        c.weight_slots = 3;
        c.mlen = 8;
        c.refinement.as_mut().unwrap().operand_latch_bytes = 3 * c.blen * c.mlen * 2;
    }
    let old = simulate(w.clone(), a.clone(), &bytes).await;
    a.diagnostic.charge_legacy_control = true;
    for ports in [1, 2] {
        a.diagnostic.control_ports = ports;
        let r = simulate(w.clone(), a.clone(), &bytes).await;
        assert_eq!(r.output_bf16, old.output_bf16);
        assert_eq!(r.output_f32, old.output_f32);
        assert_eq!(r.pre_round_output_f32, old.pre_round_output_f32);
        assert_eq!(r.hbm_read_bytes, old.hbm_read_bytes);
        for (i, c) in r.cores.iter().enumerate() {
            let d = c.refinement.as_ref().unwrap();
            let l = d.legacy_control.as_ref().unwrap();
            let tiles: u64 = c
                .projections
                .iter()
                .map(|p| p.metrics.tile_loads.count)
                .sum();
            let issues: u64 = c
                .projections
                .iter()
                .map(|p| p.metrics.tile_loads.count * p.m.div_ceil(d.m_rows) as u64)
                .sum();
            assert_eq!(l.operations_by_kind, [issues, tiles, issues]);
            assert_eq!(
                l.service_cycles_by_kind,
                [2 * issues, 3 * tiles, 2 * issues]
            );
            assert_eq!(
                l.service_ps_by_kind,
                l.service_cycles_by_kind.map(|x| x * a.clock_period_ps)
            );
            assert_eq!(
                c.accumulator_peak_bytes,
                old.cores[i].accumulator_peak_bytes + (ports - 1) * 64
            );
            let total: u64 = l.service_ps_by_kind.iter().sum();
            assert!(
                l.control_occupancy_ps <= total && total <= l.control_occupancy_ps * ports as u64
            );
        }
    }
}

#[test]
fn cohort_mask_and_control_port_costs_cannot_escape_validation() {
    let (mut w, bytes, _) = fixture();
    let mut a = event_architecture(64, 2, ReadySelection::Rotating);
    for c in &mut a.cores {
        let r = c.refinement.as_mut().unwrap();
        r.m_rows = 1;
        r.stream_ctrl.as_mut().unwrap().cohort_control = true;
    }
    for ports in [0, 3] {
        a.diagnostic.control_ports = ports;
        assert!(validate(&w, &a, bytes.len() as u64).is_err());
    }
    a.diagnostic.control_ports = 1;
    w.inputs_bf16 = vec![w.inputs_bf16[0].clone(); 33];
    w.routes = (0..33)
        .map(|token| Route {
            token,
            slot: 0,
            expert: 0,
            weight: 1.0,
        })
        .collect();
    assert!(validate(&w, &a, bytes.len() as u64).is_err());
}

#[tokio::test]
async fn long_event_fanout_backpressures_completions_without_losing_them() {
    let (mut w, bytes, dense) = fixture();
    w.inputs_bf16 = vec![w.inputs_bf16[0].clone(); 32];
    w.routes = (0..32)
        .map(|token| Route {
            token,
            slot: 0,
            expert: 0,
            weight: 1.0,
        })
        .collect();
    w.shared_expert = None;
    let mut a = event_architecture(32, 1, ReadySelection::Rotating);
    a.mac_pipeline_cycles = 0;
    for c in &mut a.cores {
        c.weight_slots = 1;
        let r = c.refinement.as_mut().unwrap();
        r.m_rows = 1;
        r.accumulator_elements_per_cycle = 32;
    }
    let report = simulate(w.clone(), a, &bytes).await;
    assert_eq!(report.output_bf16, reference(&w, &dense));
    let active = report.cores.iter().find(|c| c.jobs > 0).unwrap();
    let s = active
        .refinement
        .as_ref()
        .unwrap()
        .stream_ctrl
        .as_ref()
        .unwrap();
    assert!(s.blocked_producers_peak > 0);
    assert!(s.event_queue_wait_ps > 0);
    assert_eq!(s.event_queue_peak, 1);
    assert_eq!(s.events_enqueued, s.events_processed);
}

#[test]
fn event_control_is_disabled_by_default_and_charged_to_accumulator() {
    let (w, bytes, _) = fixture();
    let mut legacy = pool_architecture(32, 2);
    for c in &mut legacy.cores {
        c.weight_slots = 3;
    }
    let mut candidate = legacy.clone();
    for c in &mut candidate.cores {
        c.refinement.as_mut().unwrap().stream_ctrl = Some(StreamControlConfig {
            event_ready: true,
            ..Default::default()
        });
        assert_eq!(c.stream_control_bytes().unwrap(), 208);
    }
    let disabled: StreamControlConfig = serde_json::from_str("{}").unwrap();
    assert!(!disabled.event_ready);
    assert_eq!(disabled.selection, ReadySelection::Rotating);
    let c = &mut candidate.cores[0];
    let r = c.refinement.as_ref().unwrap();
    let before = 4 * w.input_dim.max(w.expert_hidden_dim) * 4
        + 4 * c.blen * (c.mlen.ilog2() as usize + candidate.mac_pipeline_cycles as usize)
        + 128 * 32
        + 64 * (3 + 2)
        + 4 * 32 * r.m_rows * c.blen;
    c.accumulator_bytes = before + 208;
    validate(&w, &candidate, bytes.len() as u64).unwrap();
    candidate.cores[0].accumulator_bytes -= 1;
    assert!(validate(&w, &candidate, bytes.len() as u64).is_err());
}

#[tokio::test]
async fn both_event_selectors_preserve_bytes_k_order_ports_and_repeat() {
    let (w, bytes, dense) = fixture();
    let legacy = simulate(w.clone(), pool_architecture(8, 2), &bytes).await;
    for selection in [ReadySelection::Lowest, ReadySelection::Rotating] {
        let a = event_architecture(8, 2, selection);
        let one = simulate(w.clone(), a.clone(), &bytes).await;
        let two = simulate(w.clone(), a.clone(), &bytes).await;
        assert_eq!(one.output_bf16, reference(&w, &dense));
        assert_eq!(one.output_f32, legacy.output_f32);
        assert_eq!(
            one.pre_round_output_f32
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            legacy
                .pre_round_output_f32
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(one.hbm_read_bytes, legacy.hbm_read_bytes);
        assert_eq!(
            serde_json::to_value(&one).unwrap(),
            serde_json::to_value(&two).unwrap()
        );
        for ((old, new), config) in legacy.cores.iter().zip(&one.cores).zip(&a.cores) {
            let detail = new.refinement.as_ref().unwrap();
            let s = detail.stream_ctrl.as_ref().unwrap();
            let pool = detail.output_pool.as_ref().unwrap();
            assert_eq!(s.events_enqueued, s.events_processed);
            assert!(s.event_queue_peak <= config.weight_slots);
            assert_eq!(s.budget, "accumulator");
            assert_eq!(
                new.accumulator_peak_bytes,
                old.accumulator_peak_bytes + config.stream_control_bytes().unwrap()
            );
            assert_eq!(new.vector_sram_peak_bytes, old.vector_sram_peak_bytes);
            assert_eq!(
                detail.accumulator_port_busy_ps,
                old.refinement.as_ref().unwrap().accumulator_port_busy_ps
            );
            assert_eq!(
                detail.weight_port_busy_ps,
                old.refinement.as_ref().unwrap().weight_port_busy_ps
            );
            assert_eq!(s.fanout_writes, pool.context_updates);
            assert_eq!(s.event_fanout_busy_ps, s.fanout_writes * a.clock_period_ps);
            assert_eq!(
                pool.scheduler_busy_ps,
                pool.scheduler_visits * a.clock_period_ps
            );
            assert!(pool.scheduler_busy_ps <= one.total_ps);
            assert!(
                new.projections
                    .iter()
                    .all(|p| p.metrics.first_tile_arrival_ps.is_some())
            );
        }
    }
}

#[tokio::test]
async fn event_queue_backpressure_drains_with_one_slot_slow_feedback_and_partial_cohorts() {
    let (w, bytes, dense) = fixture();
    for mt in [1, 3] {
        let mut a = event_architecture(8, 1, ReadySelection::Rotating);
        a.mac_pipeline_cycles = 100;
        a.global_dma_credits = 1;
        a.global_dma_staging_bytes = 64;
        for c in &mut a.cores {
            c.weight_slots = 1;
            let r = c.refinement.as_mut().unwrap();
            r.m_rows = mt;
            r.accumulator_elements_per_cycle = 1;
            r.weight_read_elements_per_cycle = 1;
        }
        let result = simulate(w.clone(), a, &bytes).await;
        assert_eq!(result.output_bf16, reference(&w, &dense));
        assert_eq!(result.job_completions.len(), 3);
        assert!(result.total_ps < 100_000_000);
        for c in &result.cores {
            let s = c.refinement.as_ref().unwrap().stream_ctrl.as_ref().unwrap();
            assert_eq!(s.event_queue_peak, 1);
            assert_eq!(s.events_enqueued, s.events_processed);
        }
    }
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

#[tokio::test]
async fn diagnostic_third_stage_needs_opt_in_and_full_weight_and_control_funding() {
    let (w, bytes, _) = fixture();
    let mut a = event_architecture(8, 3, ReadySelection::Rotating);
    for c in &mut a.cores {
        c.weight_slots = 3;
        c.weight_sram_bytes =
            3 * c.blen * c.mlen * 25 / 8 + c.refinement.as_ref().unwrap().operand_latch_bytes;
    }
    assert!(validate(&w, &a, bytes.len() as u64).is_err());
    a.diagnostic.allow_three_operand_stages = true;
    validate(&w, &a, bytes.len() as u64).unwrap();
    a.cores[0].weight_sram_bytes -= 1;
    assert!(validate(&w, &a, bytes.len() as u64).is_err());
    a.cores[0].weight_sram_bytes += 1;
    let mut ordinary = a.clone();
    for c in &mut ordinary.cores {
        let r = c.refinement.as_mut().unwrap();
        r.output_pool.as_mut().unwrap().operand_stages = 2;
        r.operand_latch_bytes = 2 * c.blen * c.mlen * 2;
    }
    let before = simulate(w.clone(), ordinary, &bytes).await;
    for (config, observed) in a.cores.iter_mut().zip(&before.cores) {
        assert!(observed.jobs > 0);
        config.accumulator_bytes = observed.accumulator_peak_bytes + 64;
    }
    validate(&w, &a, bytes.len() as u64).unwrap();
    a.cores[0].accumulator_bytes -= 1;
    assert!(validate(&w, &a, bytes.len() as u64).is_err());
}

#[tokio::test]
async fn event_cost_and_third_stage_diagnostics_preserve_k_order_bytes_and_drain() {
    let (w, bytes, _) = fixture();
    let ordinary = simulate(
        w.clone(),
        event_architecture(8, 2, ReadySelection::Rotating),
        &bytes,
    )
    .await;
    for stages in [2, 3] {
        for zero_cost in [false, true] {
            let mut a = event_architecture(8, stages, ReadySelection::Rotating);
            a.diagnostic.allow_three_operand_stages = stages == 3;
            a.diagnostic.event_updates_zero_cost = zero_cost;
            for c in &mut a.cores {
                c.weight_slots = 3;
            }
            let r = simulate(w.clone(), a.clone(), &bytes).await;
            assert_eq!(r.output_bf16, ordinary.output_bf16);
            for (got, expected) in [&r.output_f32, &r.pre_round_output_f32]
                .into_iter()
                .zip([&ordinary.output_f32, &ordinary.pre_round_output_f32])
            {
                assert_eq!(
                    got.iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(r.hbm_read_bytes, ordinary.hbm_read_bytes);
            for c in &r.cores {
                let d = c.refinement.as_ref().unwrap();
                let p = d.output_pool.as_ref().unwrap();
                let s = d.stream_ctrl.as_ref().unwrap();
                assert_eq!(s.events_enqueued, s.events_processed);
                assert_eq!(
                    s.event_counts_by_kind,
                    [p.tile_admissions, p.tile_admissions, p.context_updates]
                );
                assert_eq!(
                    s.event_update_cycles_by_kind,
                    [p.tile_admissions, p.context_updates, 2 * p.context_updates]
                );
                let busy: u64 = s.event_service_ps_by_kind.iter().sum();
                assert_eq!(
                    s.event_service_mac_overlap_ps + s.event_service_no_mac_ps,
                    busy
                );
                assert!(s.event_service_mac_overlap_ps <= c.compute_busy_ps);
                let nominal: u64 = s.event_update_cycles_by_kind.iter().sum();
                assert_eq!(
                    busy,
                    if zero_cost {
                        0
                    } else {
                        nominal * a.clock_period_ps
                    }
                );
                assert_eq!(
                    p.scheduler_busy_ps,
                    (p.scheduler_visits - if zero_cost { nominal } else { 0 }) * a.clock_period_ps
                );
            }
        }
    }
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

#[tokio::test]
async fn diagnostic_factors_preserve_values_work_and_frozen_job_ownership() {
    let (w, bytes, dense) = fixture();
    for pooled in [false, true] {
        let mut a = if pooled {
            pool_architecture(8, 2)
        } else {
            refined_architecture(2)
        };
        a.dispatch_policy = DispatchPolicy::WorkConserving;
        let base = simulate(w.clone(), a.clone(), &bytes).await;
        let mut order = vec![Vec::new(); a.cores.len()];
        let mut completions: Vec<_> = base.job_completions.iter().collect();
        completions.sort_by_key(|j| j.start_ps);
        for job in completions {
            let core = a.cores.iter().position(|c| c.id == job.core).unwrap();
            order[core].push(job.job);
        }
        a.diagnostic.fixed_job_order = Some(order.clone());
        for variant in 0..8 {
            let mut candidate = a.clone();
            let d = &mut candidate.diagnostic;
            match variant {
                1 => d.mac_speedup = 2,
                2 => d.activation_speedup = 2,
                3 => d.weight_port_speedup = 2,
                4 => d.accumulator_speedup = 2,
                5 => d.scheduler_speedup = 2,
                6 => d.vector_speedup = 2,
                7 => {
                    d.mac_speedup = 2;
                    d.activation_speedup = 2;
                    d.accumulator_speedup = 2;
                }
                _ => {}
            }
            let result = simulate(w.clone(), candidate, &bytes).await;
            assert_eq!(result.output_bf16, reference(&w, &dense));
            assert_eq!(result.useful_macs, base.useful_macs);
            assert_eq!(result.issued_macs, base.issued_macs);
            assert_eq!(result.hbm_read_bytes, base.hbm_read_bytes);
            for (index, core) in a.cores.iter().enumerate() {
                let mut jobs: Vec<_> = result
                    .job_completions
                    .iter()
                    .filter(|j| j.core == core.id)
                    .collect();
                jobs.sort_by_key(|j| j.start_ps);
                assert_eq!(jobs.iter().map(|j| j.job).collect::<Vec<_>>(), order[index]);
            }
            if variant == 6 {
                // Faster decode must not accidentally accelerate the weight SRAM port.
                for (b, c) in base.cores.iter().zip(&result.cores) {
                    assert_eq!(
                        b.refinement.as_ref().unwrap().weight_port_busy_ps,
                        c.refinement.as_ref().unwrap().weight_port_busy_ps
                    );
                }
            }
        }
        for bad in [
            vec![vec![], vec![]],
            vec![vec![0, 0], vec![1, 2]],
            vec![vec![99], vec![]],
        ] {
            a.diagnostic.fixed_job_order = Some(bad);
            assert!(validate(&w, &a, bytes.len() as u64).is_err());
        }
    }
}
