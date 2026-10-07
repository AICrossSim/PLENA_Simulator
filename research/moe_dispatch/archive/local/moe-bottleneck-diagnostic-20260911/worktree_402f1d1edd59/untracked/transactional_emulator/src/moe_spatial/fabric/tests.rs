use super::*;
#[test]
fn external_bf16_values_survive_tails_migration_and_oracle_timing() {
    let mut r = input(vec![6], &[4, 2]);
    for j in &mut r.compute.jobs {
        j.n = 7;
        j.k = 19;
    }
    let data = operands::Operands {
        jobs: r
            .compute
            .jobs
            .iter()
            .map(|j| operands::Values {
                x: (0..j.m * j.k)
                    .map(|i| bf16::from_f32(((i * 7 + 3) % 31) as f32 / 13.0 - 1.0).to_f32())
                    .collect(),
                w: (0..j.n * j.k)
                    .map(|i| bf16::from_f32(((i * 5 + j.expert) % 23) as f32 / 11.0 - 1.0).to_f32())
                    .collect(),
            })
            .collect(),
    };
    r.fabric.control = Control::TileCohort;
    let baseline = run_with_operands(&r, Some(&data)).unwrap();
    assert_ne!(baseline.output_fp32_bits, run(&r).unwrap().output_fp32_bits);
    for shape in [vec![6], vec![3, 3], vec![4, 2], vec![2, 2, 2], vec![1; 6]] {
        for mode in [Ownership::PinnedExpert, Ownership::TileStealing] {
            for oracle in [false, true] {
                r.compute.m_lanes = shape.clone();
                r.compute.ownership = mode;
                r.fabric.zero_control_time = oracle;
                r.fabric.zero_weight_time = oracle;
                let actual = run_with_operands(&r, Some(&data)).unwrap();
                assert!(actual.drained);
                assert_eq!(actual.output_fp32_bits, baseline.output_fp32_bits);
                assert_eq!(actual.numerical_bit_exact, Some(true));
                let mut timing = r.clone();
                timing.compute.verify_values = false;
                let t = run(&timing).unwrap();
                assert_eq!(actual.total_cycles, t.total_cycles);
                assert_eq!(actual.invocation_sha256, t.invocation_sha256);
                assert_eq!(actual.service_sha256, t.service_sha256);
            }
        }
    }
}

#[test]
fn external_values_reject_bad_dimensions_non_bf16_and_verification_off() {
    let mut r = input(vec![6], &[1]);
    let j = &r.compute.jobs[0];
    let mut data = operands::Operands {
        jobs: vec![operands::Values {
            x: vec![1.; j.m * j.k],
            w: vec![1.; j.n * j.k],
        }],
    };
    assert!(data.validate(&r.compute).is_ok());
    data.jobs[0].x[0] = f32::NAN;
    assert!(data.validate(&r.compute).is_err());
    data.jobs[0].x[0] = 1.00001;
    assert!(data.validate(&r.compute).is_err());
    data.jobs[0].x[0] = 1.;
    data.jobs[0].w.pop();
    assert!(data.validate(&r.compute).is_err());
    data.jobs[0].w.push(1.);
    r.compute.verify_values = false;
    assert!(run_with_operands(&r, Some(&data)).is_err());
}
#[test]
fn persistent_fifo_resumes_open_cohorts_with_reserved_credit() {
    let mut r = input(vec![1], &[9]);
    r.compute.jobs[0].n = 64;
    r.fabric.control = Control::TileCohort;
    r.fabric.selector = Selector::Fifo;
    r.fabric.descriptors = 4;
    r.fabric.ready_window = 1;
    let o = run(&r).unwrap();
    audit(&r, &o);
    assert!(o.stats.total_metadata_peak <= 4 && o.stats.cohort_descriptor_peak <= 2);
    assert!(o.stats.resume_selections > 0);
}
#[test]
fn packed_port_shares_a_beat_without_overlapping_byte_reservations() {
    let mut p = Port::new("activation", 1);
    let mut log = Vec::new();
    for id in 0..4 {
        p.reserve_data(0, 64, (192, false, true), &[id], &mut log);
    }
    assert_eq!(
        log.iter().map(|s| s.end).collect::<Vec<_>>(),
        vec![1, 1, 1, 2]
    );
    assert_eq!(
        log.iter()
            .map(|s| s.byte_start.unwrap())
            .collect::<Vec<_>>(),
        vec![0, 64, 128, 192]
    );
}
#[test]
fn packed_operands_preserve_numerics_and_local_commit_order() {
    let mut r = input(vec![1; 6], &[32, 3]);
    r.fabric.control = Control::TileCohort;
    let a = run(&r).unwrap();
    r.fabric.packed_operand_ports = true;
    let b = run(&r).unwrap();
    assert_eq!(a.output_fp32_bits, b.output_fp32_bits);
    assert!(b.drained);
    assert!(b.services.iter().any(|s| s.byte_start.is_some()));
    for c in 0..6 {
        let mut times: Vec<_> = b
            .trace
            .iter()
            .filter(|t| t.core == c)
            .map(|t| t.commit_cycle)
            .collect();
        times.sort();
        assert!(times.windows(2).all(|x| x[1] > x[0]));
    }
}
#[test]
fn persistent_tile_descriptor_covers_more_rows_than_result_capacity() {
    let mut r = input(vec![1], &[32]);
    r.compute.jobs[0].n = 2;
    r.compute.jobs[0].k = 16;
    r.fabric.control = Control::TileCohort;
    let a = run(&r).unwrap();
    audit(&r, &a);
    assert_eq!(a.stats.issue_services, 2);
    assert_eq!(a.stats.completion_services, 2);
    assert!(a.stats.total_metadata_peak <= r.fabric.descriptors);
    r.fabric.control = Control::Cohort;
    let b = run(&r).unwrap();
    assert_eq!(b.stats.issue_services, 64);
    assert_eq!(a.output_fp32_bits, b.output_fp32_bits);
    assert!(a.stats.control_service_cycles < b.stats.control_service_cycles);
}
#[test]
fn persistent_cohort_numerics_and_port_order_survive_migration() {
    for widths in [vec![6], vec![3, 3], vec![4, 2], vec![1; 6]] {
        for mode in [Ownership::PinnedExpert, Ownership::TileStealing] {
            let mut r = input(widths.clone(), &[32, 3, 1]);
            r.compute.ownership = mode;
            r.fabric.control = Control::TileCohort;
            for j in &mut r.compute.jobs {
                j.n = 7;
                j.k = 19;
            }
            let a = run(&r).unwrap();
            audit(&r, &a);
            assert_eq!(a.stats.issue_services, 36);
            assert_eq!(a.stats.completion_services, 36);
        }
    }
}
#[test]
fn long_n_queue_keeps_current_cohort_inside_the_finite_window() {
    let mut r = input(vec![1; 6], &[6]);
    r.compute.jobs[0].n = 64;
    r.compute.jobs[0].k = 8;
    let good = run(&r).unwrap();
    r.fabric.selector = Selector::Fifo;
    let fifo = run(&r).unwrap();
    assert!(good.stats.broadcast_transfers > 0);
    assert!(good.stats.source_weight_bytes < fifo.stats.source_weight_bytes);
    assert_eq!(good.output_fp32_bits, fifo.output_fp32_bits);
}
fn input(widths: Vec<usize>, ms: &[usize]) -> Input {
    Input {
        compute: Request {
            name: "fabric-test".into(),
            total_multiplier_budget: widths.iter().sum::<usize>() * 2 * 8,
            m_lanes: widths,
            n_lanes: 2,
            k_lanes: 8,
            result_latency_cycles: 5,
            issue_interval_cycles: 1,
            ownership: Ownership::TileStealing,
            verify_values: true,
            record_trace: true,
            jobs: ms
                .iter()
                .enumerate()
                .map(|(e, &m)| Job {
                    expert: e,
                    m,
                    n: 4,
                    k: 16,
                    seed: 13 + e as u32,
                })
                .collect(),
        },
        fabric: Fabric {
            weight_bpc: 32,
            activation_bpc: 64,
            accumulator_bpc: 64,
            ..Default::default()
        },
    }
}
fn audit(r: &Input, o: &Output) {
    let mut used = BTreeSet::new();
    let mut previous = BTreeMap::new();
    for t in &o.trace {
        assert!(t.admitted <= t.descriptor_ready && t.descriptor_ready <= t.activation_ready);
        assert!(
            t.issue_cycle >= t.descriptor_ready
                && t.issue_cycle >= t.weight_ready
                && t.issue_cycle >= t.activation_ready
        );
        assert_eq!(t.mac_done, t.issue_cycle + r.compute.result_latency_cycles);
        assert!(t.mac_done <= t.rmw_done && t.rmw_done <= t.commit_cycle);
        for &row in &t.rows {
            assert!(used.insert((t.expert, row, t.n_start, t.k_start)));
            if t.k_start > 0 {
                assert!(
                    previous[&(t.expert, row, t.n_start, t.k_start - r.compute.k_lanes)]
                        <= t.issue_cycle
                );
            }
            previous.insert((t.expert, row, t.n_start, t.k_start), t.commit_cycle);
        }
    }
    let mut ports: BTreeMap<(&str, usize), u64> = BTreeMap::new();
    for s in &o.services {
        assert!(s.start >= s.released && s.end >= s.start);
        let p = ports.entry((&s.resource, s.port)).or_default();
        assert!(s.start >= *p);
        *p = s.end;
    }
    assert_eq!(o.numerical_bit_exact, Some(true));
    assert!(o.drained);
}
#[test]
fn finite_ports_gate_execution_and_repeat_exactly() {
    let r = input(vec![4, 2], &[4, 2]);
    let o = run(&r).unwrap();
    audit(&r, &o);
    assert_eq!(o, run(&r).unwrap());
    assert!(o.trace.iter().any(|t| t.issue_cycle > t.admitted));
}
#[test]
fn broadcast_reads_one_payload_but_allocates_each_receiver() {
    let mut r = input(vec![1; 6], &[6]);
    r.compute.jobs[0].n = 2;
    r.compute.jobs[0].k = 8;
    let a = run(&r).unwrap();
    audit(&r, &a);
    r.fabric.broadcast = false;
    let b = run(&r).unwrap();
    audit(&r, &b);
    assert_eq!(a.stats.source_weight_bytes, 32);
    assert_eq!(a.stats.delivered_weight_bytes, 192);
    assert_eq!(b.stats.source_weight_bytes, 192);
    assert_eq!(a.stats.cache_misses, 6);
    assert_eq!(a.output_fp32_bits, b.output_fp32_bits);
}
#[test]
fn retention_serves_multiple_m_rows_without_refetch() {
    let mut r = input(vec![1], &[8]);
    r.compute.jobs[0].n = 2;
    r.compute.jobs[0].k = 8;
    r.fabric.zero_control_time = true;
    r.fabric.zero_weight_time = true;
    r.fabric.zero_activation_time = true;
    r.fabric.zero_accumulator_time = true;
    let a = run(&r).unwrap();
    r.fabric.retain_weights = false;
    let b = run(&r).unwrap();
    assert_eq!(a.stats.source_weight_bytes, 32);
    assert!(b.stats.source_weight_bytes > a.stats.source_weight_bytes);
    assert_eq!(a.output_fp32_bits, b.output_fp32_bits);
    audit(&r, &b);
}
#[test]
fn accumulator_backpressure_preserves_k_order() {
    let mut r = input(vec![2, 1], &[5, 2]);
    let fast = run(&r).unwrap();
    r.fabric.accumulator_bpc = 1;
    let slow = run(&r).unwrap();
    audit(&r, &slow);
    assert!(slow.total_cycles > fast.total_cycles);
    assert_eq!(slow.output_fp32_bits, fast.output_fp32_bits);
}
#[test]
fn activation_bandwidth_is_shared_and_causal() {
    let mut r = input(vec![2, 1], &[5, 2]);
    let fast = run(&r).unwrap();
    r.fabric.activation_bpc = 1;
    let slow = run(&r).unwrap();
    assert!(slow.total_cycles > fast.total_cycles);
    assert_eq!(slow.stats.activation_bytes, fast.stats.activation_bytes);
    audit(&r, &slow);
}
#[test]
fn tiny_credits_and_capacity_cannot_release_early_or_deadlock() {
    let mut r = input(vec![2, 1], &[5, 2]);
    r.fabric.descriptors = 1;
    r.fabric.stages_per_core = 1;
    r.fabric.slots_per_m = 1;
    let o = run(&r).unwrap();
    audit(&r, &o);
    assert_eq!(o.stats.descriptor_peak, 1);
    assert!(
        o.stats.simultaneous_operand_result_peak_bytes
            <= o.stats.weight_budget_bytes
                + o.stats.activation_budget_bytes
                + o.stats.result_budget_bytes
    );
}
#[test]
fn cohort_control_reduces_services_without_losing_work() {
    let mut r = input(vec![1; 6], &[6]);
    r.compute.jobs[0].n = 2;
    r.compute.jobs[0].k = 8;
    let a = run(&r).unwrap();
    r.fabric.control = Control::Invocation;
    let b = run(&r).unwrap();
    assert_eq!(a.stats.issue_services, 1);
    assert_eq!(b.stats.issue_services, 6);
    assert_eq!(a.stats.completion_services, 1);
    assert_eq!(b.stats.completion_services, 6);
    assert_eq!(a.output_fp32_bits, b.output_fp32_bits);
    audit(&r, &b);
}
#[test]
fn tails_pinning_stealing_selectors_and_controls_are_bit_exact() {
    let mut reference = None;
    for widths in [vec![6], vec![3, 3], vec![4, 2], vec![1; 6]] {
        for mode in [Ownership::PinnedExpert, Ownership::TileStealing] {
            for selector in [Selector::Fifo, Selector::Affinity] {
                let mut r = input(widths.clone(), &[5, 3, 1]);
                r.compute.ownership = mode;
                r.fabric.selector = selector;
                for j in &mut r.compute.jobs {
                    j.n = 7;
                    j.k = 19;
                }
                for ctl in [Control::Invocation, Control::Cohort] {
                    r.fabric.control = ctl;
                    let o = run(&r).unwrap();
                    audit(&r, &o);
                    if let Some(ref v) = reference {
                        assert_eq!(&o.output_fp32_bits, v);
                    } else {
                        reference = Some(o.output_fp32_bits.clone());
                    }
                }
            }
        }
    }
}
#[test]
fn reject_infeasible_storage_and_zero_physical_port_rate() {
    let mut r = input(vec![2], &[2]);
    r.fabric.accumulator_bytes = 1;
    assert!(run(&r).is_err());
    r.fabric.accumulator_bytes = 4096;
    r.fabric.weight_bpc = 0;
    assert!(run(&r).is_err());
    r.fabric.weight_bpc = 32;
    r.fabric.descriptors = 0;
    assert!(run(&r).is_err());
}
#[test]
fn oracle_removes_time_not_data_or_finite_ownership() {
    let mut r = input(vec![4, 2], &[4, 2]);
    let a = run(&r).unwrap();
    r.fabric.zero_control_time = true;
    r.fabric.zero_weight_time = true;
    r.fabric.zero_activation_time = true;
    r.fabric.zero_accumulator_time = true;
    let b = run(&r).unwrap();
    audit(&r, &b);
    assert!(b.total_cycles < a.total_cycles);
    assert_eq!(b.output_fp32_bits, a.output_fp32_bits);
    assert_eq!(a.stats.activation_bytes, b.stats.activation_bytes);
    assert_eq!(a.stats.accumulator_rmw_bytes, b.stats.accumulator_rmw_bytes);
    assert!(b.stats.source_weight_bytes > 0);
}

#[test]
fn live_source_completion_gates_issue_and_preserves_numerics() {
    struct Delayed {
        pending: BTreeMap<usize, u64>,
        delay: u64,
    }
    impl WeightSource for Delayed {
        fn submit(&mut self, request: WeightRead<'_>) -> Result<(), String> {
            let WeightRead { id, now, .. } = request;
            assert!(self.pending.insert(id, now + self.delay).is_none());
            Ok(())
        }
        fn advance(&mut self, now: u64) -> Result<Vec<usize>, String> {
            let ids: Vec<_> = self
                .pending
                .iter()
                .filter_map(|(&id, &t)| (t == now).then_some(id))
                .collect();
            for id in &ids {
                self.pending.remove(id);
            }
            Ok(ids)
        }
        fn drained(&self) -> bool {
            self.pending.is_empty()
        }
    }
    let mut r = input(vec![4, 2], &[4, 2]);
    r.fabric.control = Control::TileCohort;
    let base = run(&r).unwrap();
    let mut d = Delayed {
        pending: BTreeMap::new(),
        delay: 100,
    };
    let observed = run_with_source(&r, None, Some(&mut d)).unwrap();
    assert!(observed.total_cycles > base.total_cycles);
    assert_eq!(observed.output_fp32_bits, base.output_fp32_bits);
    assert!(d.drained());
    for t in &observed.trace {
        assert!(t.issue_cycle >= t.weight_ready);
        // Late multicast joiners share an earlier request's completion.
        let first = observed
            .trace
            .iter()
            .filter(|other| {
                other.expert == t.expert && other.n_start == t.n_start && other.k_start == t.k_start
            })
            .map(|other| other.descriptor_ready)
            .min()
            .unwrap();
        assert!(t.weight_ready >= first + 100);
    }
}
