use super::*;

fn request(widths: &[usize], me: &[usize], n: usize, k: usize, latency: u64) -> Request {
    Request {
        name: "unit_test".into(),
        m_lanes: widths.to_vec(),
        n_lanes: 4,
        k_lanes: 8,
        total_multiplier_budget: widths.iter().sum::<usize>() * 4 * 8,
        result_latency_cycles: latency,
        issue_interval_cycles: 1,
        ownership: Ownership::PinnedExpert,
        verify_values: true,
        record_trace: true,
        jobs: me
            .iter()
            .enumerate()
            .map(|(i, &m)| Job {
                expert: i + 19,
                m,
                n,
                k,
                seed: i as u32 * 13 + 11,
            })
            .collect(),
    }
}

#[test]
fn spatial_resource_budget_counts_m_and_rejects_temporal_budget() {
    let mut r = request(&[4, 2], &[4, 2], 4, 8, 25);
    assert_eq!(run(&r).unwrap().total_multipliers, 6 * 4 * 8);
    r.total_multiplier_budget = 2 * 4 * 8;
    assert!(run(&r).unwrap_err().contains("budget"));
    r.total_multiplier_budget = 6 * 4 * 8;
    r.m_lanes[1] = 0;
    assert!(run(&r).is_err());
    r.m_lanes[1] = 2;
    r.issue_interval_cycles = 26;
    assert!(run(&r).is_err());
}

#[test]
fn motivation_counterexample_and_no_win_controls() {
    for (me, expected) in [
        ([4, 2], [26, 26, 25]),
        ([3, 3], [26, 25, 26]),
        ([5, 1], [26, 26, 26]),
        ([2, 2], [26, 25, 25]),
    ] {
        let mut reference = None;
        for (widths, cycles) in [vec![6], vec![3, 3], vec![4, 2]].iter().zip(expected) {
            let r = request(widths, &me, 4, 8, 25);
            let out = run(&r).unwrap();
            assert_eq!(out.total_cycles, cycles, "{me:?} {widths:?}");
            assert_eq!(out.numerical_bit_exact, Some(true));
            if let Some(ref old) = reference {
                assert_eq!(&out.output_fp32_bits, old);
            }
            reference = Some(out.output_fp32_bits.clone());
            assert_eq!(out, run(&r).unwrap());
        }
    }
}

#[test]
fn latency_is_not_wave_count_and_experts_switch_without_drain() {
    let r = request(&[6], &[4, 2], 4, 8, 25);
    let out = run(&r).unwrap();
    assert_eq!(out.cores[0].issue_cycles, [0, 1]);
    assert_eq!(out.trace[0].completion_cycle, 25);
    assert_eq!(out.total_cycles, 26);
    assert_ne!(out.trace[0].expert, out.trace[1].expert);
    assert!(out.trace[1].issue_cycle < out.trace[0].completion_cycle);
    let mut r = r;
    r.issue_interval_cycles = 25;
    assert_eq!(run(&r).unwrap().total_cycles, 50);
}

#[test]
fn k_segments_wait_for_their_own_previous_commit() {
    let r = request(&[6], &[1], 1, 19, 5);
    let out = run(&r).unwrap();
    assert_eq!(out.cores[0].issue_cycles, [0, 5, 10]);
    assert_eq!(out.total_cycles, 15);
    assert_eq!(out.dependency_order_checks, 3);
    assert_eq!(out.useful_macs, 19);
    assert_eq!(out.issued_mac_slots, 3 * 6 * 4 * 8);
}

#[test]
fn tail_arithmetic_and_migration_preserve_values_and_lifetimes() {
    let mut reference = None;
    for widths in [vec![6], vec![3, 3], vec![4, 2], vec![1; 6]] {
        for mode in [Ownership::PinnedExpert, Ownership::TileStealing] {
            let mut r = request(&widths, &[5, 3, 1], 7, 19, 9);
            r.ownership = mode;
            let out = run(&r).unwrap();
            assert_eq!(out.useful_macs, 9 * 7 * 19);
            assert!(out.tail_mac_slots > 0 && out.requests_drained);
            assert!(
                out.cores
                    .iter()
                    .all(|c| c.pipeline_peak <= c.pipeline_capacity)
            );
            if let Some(ref old) = reference {
                assert_eq!(&out.output_fp32_bits, old);
            }
            reference = Some(out.output_fp32_bits.clone());
            let mut timing = r.clone();
            timing.verify_values = false;
            let timed = run(&timing).unwrap();
            assert_eq!(timed.total_cycles, out.total_cycles);
            assert_eq!(timed.trace, out.trace);
            assert_eq!(timed.cores, out.cores);
            assert_eq!(timed.numerical_bit_exact, None);
        }
    }
}

#[test]
fn small_uniform_engines_are_a_strong_compute_only_challenger() {
    let mut r = request(&[1; 6], &[4, 2], 4, 8, 25);
    r.ownership = Ownership::TileStealing;
    let out = run(&r).unwrap();
    assert_eq!(out.total_cycles, 25);
    assert_eq!(out.tail_mac_slots, 0);
    assert_eq!(out.cores.iter().filter(|c| c.invocations > 0).count(), 6);
}

#[test]
fn pending_register_capacity_is_finite_and_latency_can_be_hidden() {
    let r = request(&[4, 2], &[4, 2], 128, 32, 5);
    let out = run(&r).unwrap();
    assert_eq!(out.cores[0].pipeline_peak, 5);
    assert_eq!(out.cores[1].pipeline_peak, 5);
    assert_eq!(out.cores[0].invocations, 128);
    assert_eq!(out.total_cycles, 128 + 5 - 1);
    assert_eq!(
        out.cores
            .iter()
            .map(|c| c.pipeline_result_register_bytes)
            .sum::<usize>(),
        5 * 6 * 4 * 4
    );
}

#[test]
fn pairwise_reduction_is_explicit_not_a_scalar_sum_in_disguise() {
    let values = vec![1e20, 1.0, -1e20, 1.0];
    let sequential = values.iter().copied().fold(0.0, |a, b| a + b);
    assert_eq!(sequential, 1.0);
    assert_eq!(tree_reduce(values), 0.0);
}
