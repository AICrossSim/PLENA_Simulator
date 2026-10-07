use super::tests::{fixture, reference, refined_architecture, simulate};
use super::*;

fn configuration(window: usize, aging: Option<u64>) -> Architecture {
    let mut a = refined_architecture(2);
    for c in &mut a.cores {
        c.accumulator_bytes = 32_768;
        c.weight_slots = 3;
        c.mlen = 8;
        let r = c.refinement.as_mut().unwrap();
        r.operand_latch_bytes = 2 * c.blen * c.mlen * 2;
        r.output_pool = Some(OutputPoolConfig {
            output_contexts: 32,
            operand_stages: 2,
            scheduler_cycles: 1,
        });
        r.stream_ctrl = Some(StreamControlConfig {
            event_ready: true,
            cohort_control: true,
            split_slot_lifetime: true,
            split_window: Some(SplitWindowConfig {
                window_tiles: window,
                load_latency_sum_ps: 100_000,
                load_latency_samples: 1,
                aging_multiplier: aging,
            }),
            ..Default::default()
        });
    }
    a
}

fn check_lifetimes(r: &RunReport, a: &Architecture) {
    for (core, cfg) in r.cores.iter().zip(&a.cores) {
        let d = core.refinement.as_ref().unwrap();
        let s = d.split_window.as_ref().unwrap();
        assert_eq!(s.live_current_bytes, [0; 3]);
        let mut last = 0;
        let mut peak = 0;
        for &[at, packed, operand, decode] in &s.lifetime_changes {
            assert!(at >= last);
            last = at;
            assert_eq!(packed % s.packed_slot_bytes as u64, 0);
            assert_eq!(operand % s.operand_stage_bytes as u64, 0);
            assert_eq!(decode % s.operand_stage_bytes as u64, 0);
            assert!(packed <= (s.packed_slots * s.packed_slot_bytes) as u64);
            assert!(operand + decode <= (2 * s.operand_stage_bytes) as u64);
            assert!(packed + operand + decode <= cfg.weight_sram_bytes as u64);
            peak = peak.max(packed + operand + decode);
        }
        assert_eq!(peak, s.live_peak_bytes);
        assert_eq!(s.live_peak_classes.iter().sum::<u64>(), peak);
        assert!(s.window_max_candidates <= s.packed_slots);
        let p = d.output_pool.as_ref().unwrap();
        let ctrl = d.stream_ctrl.as_ref().unwrap();
        let admission: u64 = core
            .projections
            .iter()
            .map(|p| p.metrics.band_admissions * (p.m.div_ceil(d.m_rows) as u64 + 1))
            .sum();
        assert_eq!(
            p.scheduler_visits,
            admission
                + 10 * p.tile_admissions
                + p.context_updates
                + 4 * ctrl.burst_starts
                + ctrl.completion_mask_cycles
        );
        assert_eq!(s.packed_arrivals.count, p.tile_admissions);
        assert_eq!(s.window_selections, p.tile_admissions);
        assert_eq!(s.window_header_updates, p.tile_admissions);
        assert_eq!(s.window_header_bytes, 16 * s.packed_slots);
        assert_eq!(ctrl.writeback_order_checks, p.context_updates);
    }
}

#[tokio::test]
async fn split_lifetimes_keep_exact_tail_arithmetic_bytes_and_all_finite_stages() {
    for (me, mt, feedback) in [(1, 1, 0), (8, 1, 60), (17, 1, 0), (32, 4, 60)] {
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
        let mut old = configuration(3, None);
        old.dispatch_threshold = 1;
        old.mac_pipeline_cycles = feedback;
        for c in &mut old.cores {
            let r = c.refinement.as_mut().unwrap();
            r.m_rows = mt;
            let s = r.stream_ctrl.as_mut().unwrap();
            s.split_slot_lifetime = false;
            s.split_window = None;
        }
        let baseline = simulate(w.clone(), old, &bytes).await;
        for window in [3, 6] {
            for aging in [None, Some(2), Some(4), Some(8)] {
                let mut a = configuration(window, aging);
                a.dispatch_threshold = 1;
                a.mac_pipeline_cycles = feedback;
                for c in &mut a.cores {
                    c.refinement.as_mut().unwrap().m_rows = mt;
                }
                let one = simulate(w.clone(), a.clone(), &bytes).await;
                let two = simulate(w.clone(), a.clone(), &bytes).await;
                assert_eq!(
                    serde_json::to_value(&one).unwrap(),
                    serde_json::to_value(&two).unwrap()
                );
                assert_eq!(one.output_bf16, reference(&w, &dense));
                for (got, want) in [&one.output_f32, &one.pre_round_output_f32]
                    .into_iter()
                    .zip([&baseline.output_f32, &baseline.pre_round_output_f32])
                {
                    assert_eq!(
                        got.iter()
                            .flatten()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>(),
                        want.iter()
                            .flatten()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
                assert_eq!(one.hbm_read_bytes, baseline.hbm_read_bytes);
                assert_eq!(one.issued_macs, baseline.issued_macs);
                check_lifetimes(&one, &a);
            }
        }
    }
}

#[test]
fn split_configuration_rejects_unpaid_buffers_ports_and_invalid_calibration() {
    let (w, bytes, _) = fixture();
    let good = configuration(6, Some(4));
    validate(&w, &good, bytes.len() as u64).unwrap();
    for fault in 0..8 {
        let mut a = good.clone();
        let c = &mut a.cores[0];
        let r = c.refinement.as_mut().unwrap();
        match fault {
            0 => c.weight_sram_bytes = 6 * c.blen * (c.mlen / 8) * 9 + r.operand_latch_bytes - 1,
            1 => {
                r.stream_ctrl
                    .as_mut()
                    .unwrap()
                    .split_window
                    .as_mut()
                    .unwrap()
                    .load_latency_samples = 0
            }
            2 => {
                r.stream_ctrl
                    .as_mut()
                    .unwrap()
                    .split_window
                    .as_mut()
                    .unwrap()
                    .window_tiles = 7
            }
            3 => {
                r.stream_ctrl
                    .as_mut()
                    .unwrap()
                    .split_window
                    .as_mut()
                    .unwrap()
                    .aging_multiplier = Some(3)
            }
            4 => r.stream_ctrl.as_mut().unwrap().split_window = None,
            5 => r.stream_ctrl.as_mut().unwrap().cohort_control = false,
            6 => r.output_pool.as_mut().unwrap().operand_stages = 1,
            7 => r.output_pool = None,
            _ => unreachable!(),
        }
        assert!(
            validate(&w, &a, bytes.len() as u64).is_err(),
            "fault {fault}"
        );
    }
    let s = SplitWindowConfig {
        window_tiles: 6,
        load_latency_sum_ps: 81_069_000,
        load_latency_samples: 768,
        aging_multiplier: Some(4),
    };
    assert_eq!(s.threshold_ps().unwrap(), Some(422_235));
}

#[tokio::test]
async fn native_split_reserved_credits_preserve_outputs_and_drain_every_reservation() {
    use memory::{MemoryBacked, WithStats, WithTiming};
    use std::sync::Arc;
    let (w, bytes, dense) = fixture();
    let mut previous = None;
    for (window, weighted) in [(3, false), (6, true)] {
        let mut a = configuration(window, Some(4));
        a.global_dma_credits = 16;
        a.global_dma_staging_bytes = 16 * 64;
        a.dma = Some(DmaConfig {
            issue_policy: ramulator::model::IssuePolicy::PerChannel,
            sector_reads: true,
            coalesce: false,
            fair_credits: false,
            reserved_byte_credits: weighted,
            lookup_ii_cycles: 1,
            frontend_sram_bytes: 45_056,
        });
        let backing = MemoryBacked::with_capacity(bytes.len());
        backing.with_data(|dst| dst.copy_from_slice(&bytes));
        let memory = Arc::new(WithStats::new(WithTiming::new(
            ramulator::Ramulator::hbm2_preset(8).unwrap(),
            backing,
        )));
        let r = run(w.clone(), a.clone(), memory, bytes.len() as u64)
            .await
            .unwrap();
        assert_eq!(r.output_bf16, reference(&w, &dense));
        check_lifetimes(&r, &a);
        if let Some(old) = previous {
            assert_eq!(r.hbm_read_bytes, old);
        }
        previous = Some(r.hbm_read_bytes);
    }
}
