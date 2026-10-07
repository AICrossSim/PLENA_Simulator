use super::*;
fn hw() -> Hardware {
    Hardware {
        parallel_k: 512,
        mul_latency: 2,
        add_latency: 2,
        core_period_ps: 1000,
        weight_banks_total: 64,
        activation_banks_per_m: 4,
        accumulator_banks_per_m: 2,
        bank_word_bytes: 16,
        bank_read_latency: 2,
        weight_bytes_total: 48 * 1024,
        activation_bytes_total: 12 * 1024,
        accumulator_bytes_total: 2 * 1024 * 1024,
        x_slots_per_core: 2,
        result_contexts_per_core: 8,
        band_window: 16,
        control_word_bytes: 16,
        descriptor_bytes: 32,
        source: "compute_oracle".into(),
        assignment: "experts".into(),
        native: Default::default(),
        record_trace: true,
        x_reuse_bands: 0,
    }
}
fn request(m: Vec<usize>, rows: &[usize], n: usize, k: usize) -> Request {
    Request {
        name: "fixture".into(),
        m_lanes: m,
        n_lanes: 4,
        k_lanes: 512,
        total_multiplier_budget: 12288,
        result_latency_cycles: 25,
        issue_interval_cycles: 1,
        ownership: moe_spatial::Ownership::TileStealing,
        verify_values: true,
        record_trace: true,
        jobs: rows
            .iter()
            .enumerate()
            .map(|(expert, &m)| moe_spatial::Job {
                expert,
                m,
                n,
                k,
                seed: 1,
            })
            .collect(),
    }
}
#[test]
fn one_rw_banks_conflicts_and_latency() {
    let mut b = Banks::new(2, 8, 3);
    assert_eq!(b.access(0, &[(0, 16)], true), 3);
    assert_eq!(b.access(0, &[(16, 16)], false), 2); // first commands at t0, second at t1
    assert_eq!(b.access(2, &[(0, 8), (16, 8)], true), 6); // two words in bank 0
    assert_eq!(b.word_services, 6);
}
#[test]
fn rmw_holds_only_touched_bank() {
    let mut b = Banks::new(2, 8, 2);
    assert_eq!(b.rmw(0, &[(0, 8)], 3), 6);
    assert_eq!(b.access(0, &[(8, 8)], true), 2);
    assert_eq!(b.access(0, &[(0, 8)], true), 8);
}
#[test]
fn positive_and_negative_shapes() {
    let h = hw();
    let good = run(&request(vec![4, 2], &[4, 2], 128, 512), &h, None).unwrap();
    let homo = run(&request(vec![3, 3], &[4, 2], 128, 512), &h, None).unwrap();
    assert!(good.cycles < homo.cycles);
    let good = run(&request(vec![4, 2], &[3, 3], 128, 512), &h, None).unwrap();
    let homo = run(&request(vec![3, 3], &[3, 3], 128, 512), &h, None).unwrap();
    assert!(homo.cycles < good.cycles);
}
#[test]
fn k_tail_numeric_dependency_and_finite_capacity() {
    let r = request(vec![4, 2], &[7, 3], 7, 1031);
    let values = Operands {
        jobs: r
            .jobs
            .iter()
            .map(|j| moe_spatial::operands::Values {
                x: (0..j.m * j.k).map(|i| (i % 5) as f32 / 4.).collect(),
                w: (0..j.n * j.k).map(|i| (i % 7) as f32 / 8.).collect(),
            })
            .collect(),
    };
    for q in [16, 128, 512] {
        let mut h = hw();
        h.parallel_k = q;
        h.source = "onchip_oracle".into();
        let out = run(&r, &h, Some(&values)).unwrap();
        assert_eq!(out.numerical_bit_exact, Some(true));
        assert!(out.drained);
        for c in &out.cores {
            assert!(c.weight_peak_bytes <= 24 * 1024);
            assert!(c.result_context_peak <= 8);
        }
    }
}
#[test]
fn no_free_resource_from_folding() {
    let r = request(vec![6], &[4], 4, 512);
    let mut h = hw();
    let full = run(&r, &h, None).unwrap();
    h.parallel_k = 16;
    let folded = run(&r, &h, None).unwrap();
    assert_eq!(full.resource_bill["physical_multipliers_total"], 12288);
    assert_eq!(folded.resource_bill["physical_multipliers_total"], 384);
    assert!(folded.cycles > full.cycles);
}
#[test]
fn preserves_internal_consistency_and_repeats() {
    let r = request(vec![3, 3], &[1, 8, 2], 24, 1024);
    let mut h = hw();
    h.source = "onchip_oracle".into();
    h.assignment = "bands".into();
    let a = run(&r, &h, None).unwrap();
    let b = run(&r, &h, None).unwrap();
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
    for c in a.cores {
        assert_eq!(c.issue_states.values().sum::<u64>(), a.cycles);
    }
}

#[test]
fn bank_calendar_matches_independent_cycle_allocator() {
    use std::collections::BTreeSet;
    let mut banks = Banks::new(4, 8, 3);
    let mut occupied = vec![BTreeSet::new(); 4];
    let mut seed = 7_u64;
    for now in 0..150_u64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let address = (seed as usize % 32) * 8;
        let bytes = ((seed >> 32) as usize % 5 + 1) * 8;
        let read = seed & 1 != 0;
        let mut expected = now;
        for word in address / 8..(address + bytes) / 8 {
            let b = word % 4;
            // Explicit command-cycle occupancy, not the max-plus port formula.
            let mut tick = now;
            while occupied[b].contains(&tick) {
                tick += 1;
            }
            occupied[b].insert(tick);
            expected = expected.max(tick + 1 + if read { 2 } else { 0 });
        }
        assert_eq!(banks.access(now, &[(address, bytes)], read), expected);
    }
}

#[test]
fn streaming_x_does_not_need_an_uncharged_tile_buffer() {
    let mut banks = Banks::new(2, 8, 2);
    let mut bus = StreamBus {
        cycle: 0,
        used: 0,
        lanes: 2,
    };
    assert_eq!(banks.stream_write(0, &[(0, 32)], &mut bus), 3);
    assert_eq!(banks.access(3, &[(0, 32)], true), 6);
    assert_eq!(banks.stream_write(3, &[(32, 16)], &mut bus), 6);
}

#[test]
fn bounded_x_reuse_explains_a_working_set_threshold() {
    let mut h = hw();
    h.source = "onchip_oracle".into();
    h.assignment = "bands".into();
    let single = run(&request(vec![6], &[8], 64, 512), &h, None).unwrap();
    let dual = run(&request(vec![3, 3], &[8], 64, 512), &h, None).unwrap();
    // Two single-core stages hold 6+2 rows; two 3-row stages cannot hold 8 rows.
    // This verifies the declared FIFO/tag policy, not optimal cache replacement.
    assert_eq!(single.activation_input_bytes, 8 * 512 * 2);
    assert!(dual.activation_input_bytes > 2 * single.activation_input_bytes);
    assert_eq!(single.source_bytes, dual.source_bytes);
}

#[test]
fn resident_group_reuses_x_without_rereading_weights() {
    for lanes in [vec![6], vec![3, 3], vec![4, 2]] {
        let r = request(lanes, &[8], 64, 512);
        let mut h = hw();
        h.source = "onchip_oracle".into();
        h.assignment = "bands".into();
        let before = run(&r, &h, None).unwrap();
        h.x_reuse_bands = 4;
        let after = run(&r, &h, None).unwrap();
        assert_eq!(before.source_bytes, after.source_bytes);
        assert_eq!(before.useful_macs, after.useful_macs);
        assert_eq!(before.padded_macs, after.padded_macs);
        assert!(after.activation_input_bytes <= before.activation_input_bytes);
        if r.m_lanes == [3, 3] {
            // 16 bands / 4 resident bands = 4 groups; every group needs the
            // eight X rows once. No payload is shared across private cores.
            assert_eq!(after.activation_input_bytes, 4 * 8 * 512 * 2);
            assert!(after.activation_input_bytes < before.activation_input_bytes);
        }
        assert!(
            after.resource_bill["controller_state_bytes"]
                .as_u64()
                .unwrap()
                <= 4096
        );
        for (c, &m) in after.cores.iter().zip(&r.m_lanes) {
            assert!(c.activation_peak_bytes <= 2 * m * 512 * 2);
            assert!(c.weight_peak_bytes <= 40 * 1024 / r.m_lanes.len());
        }
    }
}

#[test]
fn reuse_tail_expert_boundaries_folded_k_and_order_are_bit_exact() {
    for lanes in [vec![6], vec![3, 3], vec![4, 2]] {
        let r = request(lanes, &[1, 8, 17], 19, 1031);
        let values = Operands {
            jobs: r
                .jobs
                .iter()
                .map(|j| moe_spatial::operands::Values {
                    x: (0..j.m * j.k)
                        .map(|i| ((i * 7 % 17) as f32 - 8.) / 8.)
                        .collect(),
                    w: (0..j.n * j.k)
                        .map(|i| ((i * 3 % 13) as f32 - 6.) / 16.)
                        .collect(),
                })
                .collect(),
        };
        for pk in [128, 512] {
            let mut h = hw();
            h.source = "onchip_oracle".into();
            h.assignment = "bands".into();
            h.parallel_k = pk;
            let before = run(&r, &h, Some(&values)).unwrap();
            for group in [1, 2, 4] {
                h.x_reuse_bands = group;
                let after = run(&r, &h, Some(&values)).unwrap();
                assert_eq!(after.output_fp32_bits, before.output_fp32_bits);
                assert_eq!(after.source_bytes, before.source_bytes);
                assert!(after.ownership_and_k_order_verified && after.drained);
                assert_eq!(after.numerical_bit_exact, Some(true));
                assert!(after.cores.iter().all(|c| c.result_context_peak <= 8));
            }
        }
    }
}

#[test]
fn reuse_requires_physical_weight_residency() {
    let mut h = hw();
    h.x_reuse_bands = 6;
    assert!(
        run(&request(vec![4, 2], &[8], 32, 512), &h, None)
            .err()
            .unwrap()
            .contains("resident weight slots")
    );
}

#[test]
fn reuse_preserves_descriptor_order_and_groups_only_matching_inputs() {
    let r = request(vec![3, 3], &[1, 8, 17], 19, 1031);
    let mut h = hw();
    h.assignment = "bands".into();
    let bands = map_bands(&r, &h).unwrap();
    for c in 0..2 {
        let order = tile_order(&r, &h, &bands, c);
        let plan = group_tiles(
            order.clone(),
            4,
            &bands,
            &r.jobs,
            r.m_lanes[c],
            h.x_slots_per_core,
        );
        assert_eq!(
            plan.iter().map(|p| (p.band, p.kt)).collect::<VecDeque<_>>(),
            order
        );
        let mut i = 0;
        while i < plan.len() {
            let len = plan[i].group_len;
            assert!((1..=4).contains(&len));
            for p in plan.iter().skip(i).take(len) {
                assert_eq!(bands[p.band].job, bands[plan[i].band].job);
                assert_eq!(p.kt, plan[i].kt);
                assert_eq!(p.group_len, len);
            }
            i += len;
        }
    }
}

#[test]
fn cache_fitting_experts_preserve_frozen_timing_and_bank_services() {
    for lanes in [vec![6], vec![3, 3], vec![4, 2]] {
        let r = request(lanes, &[1, 2, 4], 19, 1031);
        let mut h = hw();
        h.assignment = "bands".into();
        h.source = "onchip_oracle".into();
        let old = run(&r, &h, None).unwrap();
        h.x_reuse_bands = 4;
        let new = run(&r, &h, None).unwrap();
        assert_eq!(old.cycles, new.cycles);
        assert_eq!(old.activation_input_bytes, new.activation_input_bytes);
        assert_eq!(old.trace, new.trace);
        assert_eq!(
            serde_json::to_value(&old.weight_banks).unwrap(),
            serde_json::to_value(&new.weight_banks).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&old.activation_banks).unwrap(),
            serde_json::to_value(&new.activation_banks).unwrap()
        );
    }
}
