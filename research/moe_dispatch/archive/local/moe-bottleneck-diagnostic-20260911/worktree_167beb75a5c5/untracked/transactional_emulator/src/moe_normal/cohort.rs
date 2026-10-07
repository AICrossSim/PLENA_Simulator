//! Tile/cohort issue sequencer and a single masked completion per burst.
//! A burst has at most C<=32 issued records and one retry pass. No weight is
//! released or reloaded until every M consumer of its native tile has issued.
use super::*;

pub(super) fn mask(c: usize) -> u32 {
    assert!((1..=32).contains(&c));
    u32::MAX >> (32 - c)
}

impl Pool {
    pub(super) async fn complete_burst(
        &self,
        core: &CoreState,
        a: &Architecture,
        event: Event,
        k: usize,
    ) -> Result<(), String> {
        let bi = event.band as usize;
        let count = event.mask.count_ones() as usize;
        assert!(count > 0 && event.mask & !mask(self.mb) == 0);
        {
            let _port = self
                .access(
                    core,
                    a,
                    2,
                    false,
                    Some(2),
                    bi,
                    ControlCostClass::BurstComplete,
                )
                .await;
            let mut s = self.state.lock().unwrap();
            s.pending -= count;
            let b = s.bands[bi].as_mut().unwrap();
            assert!(!b.completing);
            b.completing = true;
            b.pending -= count;
            b.remaining -= count;
        }
        // Width, not population count: even a sparse mask traverses ceil(C/8)
        // bytes. Each visit publishes at most eight previous-K bits.
        for first in (0..self.mb).step_by(8) {
            let _port = self
                .access(core, a, 1, false, Some(2), bi, ControlCostClass::Mask)
                .await;
            let mut s = self.state.lock().unwrap();
            for mi in first..(first + 8).min(self.mb) {
                if event.mask & (1 << mi) == 0 {
                    continue;
                }
                let ci = bi * self.mb + mi;
                assert!(!s.prev_done.get(ci), "completion published twice");
                s.prev_done.set(ci, true);
                let ready = s.decoded.get(ci)
                    && s.bands[bi].as_ref().unwrap().unissued_mask & (1 << mi) != 0;
                s.ready.set(ci, ready);
            }
            if first + 8 >= self.mb {
                let b = s.bands[bi].as_mut().unwrap();
                b.completing = false;
                if b.remaining == 0 && b.slot.is_none() && b.operand_stage.is_none() {
                    assert!(b.k >= k && b.pending == 0);
                    s.bands[bi] = None;
                    s.free_bands.push_back(bi);
                    s.live -= 1;
                } else {
                    Self::queue_load(&mut s, bi, self.mb, k);
                }
            }
            drop(s);
            self.issuer.notify_one();
        }
        let mut report = core.report.lock().unwrap();
        let r = report
            .refinement
            .as_mut()
            .unwrap()
            .stream_ctrl
            .as_mut()
            .unwrap();
        r.burst_completions += 1;
        r.completed_contexts += count as u64;
        r.completion_mask_cycles += self.mb.div_ceil(8) as u64;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn cohort_issue_actor(
        self: &Arc<Self>,
        input: &[bf16],
        acc: &mut [f32],
        m: usize,
        region: &MatrixRegion,
        core: &Arc<CoreState>,
        a: &Architecture,
    ) -> Result<(), String> {
        let c = &core.config;
        let r = c.refinement.as_ref().unwrap();
        let mode = r.stream_ctrl.as_ref().unwrap().selection;
        let (n, k) = (region.rows, region.cols);
        let ex = Executor::current();
        let mut last_band = None;
        loop {
            let notification = self.issuer.notified();
            tokio::pin!(notification);
            notification.as_mut().enable();
            let selected = {
                let s = self.state.lock().unwrap();
                if s.done {
                    return Ok(());
                }
                s.ready.select(s.cursor, mode)
            };
            let Some(ci) = selected else {
                let (feedback, drain) = {
                    let s = self.state.lock().unwrap();
                    (
                        s.pending > 0,
                        s.next_n >= n && s.bands.iter().flatten().all(|b| b.k >= k),
                    )
                };
                let begin = ex.now().as_picos();
                notification.await;
                let elapsed = ex.now().as_picos() - begin;
                let mut report = core.report.lock().unwrap();
                if drain {
                    report.pipeline_drain_ps += elapsed;
                } else if feedback {
                    report.accumulator_dependency_stall_ps += elapsed;
                    report.refinement.as_mut().unwrap().output_context_stall_ps += elapsed;
                } else {
                    report.weight_ready_wait_ps += elapsed;
                }
                continue;
            };
            let bi = ci / self.mb;
            let (slot, stage, n0, k0, lifetime, attempts) = {
                let _port = self
                    .access(core, a, 2, true, None, bi, ControlCostClass::BurstStart)
                    .await;
                let s = self.state.lock().unwrap();
                let b = s.bands[bi].as_ref().unwrap();
                let slot = b.slot;
                let stage = if self.split {
                    b.operand_stage.unwrap()
                } else {
                    s.slots[slot.unwrap()].as_ref().unwrap().stage.unwrap()
                };
                (slot, stage, b.n, b.k, b.lifetime, b.unissued_mask)
            };
            let collectors = self.completion_collectors.fetch_add(1, Ordering::SeqCst) + 1;
            {
                let mut report = core.report.lock().unwrap();
                let r = report
                    .refinement
                    .as_mut()
                    .unwrap()
                    .stream_ctrl
                    .as_mut()
                    .unwrap();
                r.burst_starts += 1;
                r.completion_collectors_peak = r.completion_collectors_peak.max(collectors);
            }
            // The last receiver is a reference to one of the existing pending
            // context records. Superseded receivers are dropped, not queued.
            let mut last_done = None;
            let mut issued_mask = 0u32;
            let mut retry = 0u32;
            let started = Arc::new(AtomicUsize::new(0)); // order observers only
            let completed = Arc::new(AtomicUsize::new(0));
            for pass in 0..2 {
                let visit_mask = if pass == 0 { attempts } else { retry };
                for mi in 0..self.mb {
                    if visit_mask & (1 << mi) == 0 {
                        continue;
                    }
                    // A real sequencer cycle for every attempted block,
                    // including skips and retries. No descriptor-port access.
                    ex.resolve_at(Duration::from_picos(a.clock_period_ps)).await;
                    let ci = bi * self.mb + mi;
                    let ready = {
                        let s = self.state.lock().unwrap();
                        s.prev_done.get(ci)
                            && s.decoded.get(ci)
                            && s.bands[bi].as_ref().unwrap().unissued_mask & (1 << mi) != 0
                    };
                    {
                        let mut report = core.report.lock().unwrap();
                        let r = report
                            .refinement
                            .as_mut()
                            .unwrap()
                            .stream_ctrl
                            .as_mut()
                            .unwrap();
                        r.sequencer_cycles += 1;
                        r.sequencer_busy_ps += a.clock_period_ps;
                        r.burst_retry_checks += u64::from(pass == 1);
                        r.burst_not_ready_checks += u64::from(!ready);
                    }
                    if !ready {
                        if pass == 0 {
                            retry |= 1 << mi;
                        }
                        continue;
                    }
                    {
                        let mut s = self.state.lock().unwrap();
                        s.ready.set(ci, false);
                        s.decoded.set(ci, false);
                        s.prev_done.set(ci, false);
                        s.pending += 1;
                        let b = s.bands[bi].as_mut().unwrap();
                        assert_eq!((b.n, b.k, b.lifetime), (n0, k0, lifetime));
                        b.unissued_mask &= !(1 << mi);
                        b.unissued -= 1;
                        b.pending += 1;
                    }
                    let m0 = mi * r.m_rows;
                    let (mr, nr, kr) =
                        (r.m_rows.min(m - m0), c.blen.min(n - n0), c.mlen.min(k - k0));
                    if k0 > 0 {
                        refined_port_work(core, false, mr * nr, a.clock_period_ps).await;
                    }
                    {
                        let s = self.state.lock().unwrap();
                        for row in 0..mr {
                            for col in 0..nr {
                                let target = (m0 + row) * n + n0 + col;
                                for kk in 0..kr {
                                    let product = input[(m0 + row) * k + k0 + kk].to_f32()
                                        * s.stages[stage].operands[col * c.mlen + kk].to_f32();
                                    acc[target] += product;
                                }
                            }
                        }
                    }
                    let rows = match r.tail_policy {
                        TailPolicy::Padded => r.m_rows,
                        TailPolicy::ValidRows => mr,
                    };
                    let service = issue_service_ps(a, c, rows);
                    self.observation
                        .lock()
                        .unwrap()
                        .mac_service
                        .push((ex.now().as_picos(), ex.now().as_picos() + service));
                    let result_time = ex.now() + Duration::from_picos(service + feedback_ps(a, c));
                    let ordinal = issued_mask.count_ones() as usize;
                    issued_mask |= 1 << mi;
                    let (tx, rx) = oneshot::channel();
                    last_done = Some(rx);
                    let completion_core = core.clone();
                    let clock = a.clock_period_ps;
                    let (started, completed) = (started.clone(), completed.clone());
                    ex.spawn(async move {
                        Executor::current().resolve_at(result_time).await;
                        // Increasing result_time preserves arrival order at
                        // the existing FIFO accumulator port. Assert both
                        // enqueue and completion order without controlling it.
                        assert_eq!(started.fetch_add(1, Ordering::SeqCst), ordinal);
                        refined_port_work(&completion_core, false, mr * nr, clock).await;
                        assert_eq!(completed.fetch_add(1, Ordering::SeqCst), ordinal);
                        completion_core
                            .report
                            .lock()
                            .unwrap()
                            .refinement
                            .as_mut()
                            .unwrap()
                            .stream_ctrl
                            .as_mut()
                            .unwrap()
                            .writeback_order_checks += 1;
                        let _ = tx.send(());
                    });
                    {
                        let mut report = core.report.lock().unwrap();
                        report.useful_macs += (mr * nr * kr) as u64;
                        report.issued_macs += (rows * c.blen * c.mlen) as u64;
                        report.compute_busy_ps += service;
                        let d = report.refinement.as_mut().unwrap();
                        if last_band.is_some_and(|last| last != bi) {
                            d.independent_output_switches += 1;
                        }
                        d.output_pool.as_mut().unwrap().context_updates += 1;
                        d.stream_ctrl.as_mut().unwrap().issued_contexts += 1;
                    }
                    last_band = Some(bi);
                    self.update_peaks(core);
                    ex.resolve_at(Duration::from_picos(service)).await;
                }
            }
            assert!(
                issued_mask != 0,
                "ready burst must issue at least its selected block"
            );
            if issued_mask != attempts {
                core.report
                    .lock()
                    .unwrap()
                    .refinement
                    .as_mut()
                    .unwrap()
                    .stream_ctrl
                    .as_mut()
                    .unwrap()
                    .burst_partial += 1;
            }
            let done = last_done.unwrap();
            let (completion_pool, completion_core) = (self.clone(), core.clone());
            ex.spawn(async move {
                done.await.expect("last burst writeback must complete");
                assert_eq!(
                    completed.load(Ordering::SeqCst),
                    issued_mask.count_ones() as usize
                );
                completion_pool
                    .send(
                        &completion_core,
                        Event {
                            kind: 2,
                            source: 0,
                            band: bi as u16,
                            mask: issued_mask,
                            lifetime,
                        },
                    )
                    .await;
                completion_pool
                    .completion_collectors
                    .fetch_sub(1, Ordering::SeqCst);
            });
            {
                let mut s = self.state.lock().unwrap();
                s.cursor = ((bi + 1) * self.mb) % s.ready.len;
            }
            if self.state.lock().unwrap().bands[bi]
                .as_ref()
                .unwrap()
                .unissued
                == 0
            {
                let _port = self
                    .access(
                        core,
                        a,
                        if self.split { 2 } else { 3 },
                        false,
                        None,
                        bi,
                        ControlCostClass::OperandRetire,
                    )
                    .await;
                let mut s = self.state.lock().unwrap();
                if self.split {
                    assert!(s.bands[bi].as_ref().unwrap().slot.is_none());
                    core.change_lifetime(0, -((c.blen * c.mlen * 2) as i64), 0);
                } else {
                    s.slots[slot.unwrap()].take();
                    s.free_slots.push_back(slot.unwrap());
                }
                s.stages[stage].slot = None;
                s.free_stages.push_back(stage);
                let b = s.bands[bi].as_mut().unwrap();
                b.slot = None;
                b.operand_stage = None;
                b.k += c.mlen;
                b.unissued = self.mb;
                b.unissued_mask = mask(self.mb);
                if b.remaining == 0 && !b.completing {
                    assert!(b.k >= k && b.pending == 0);
                    s.bands[bi] = None;
                    s.free_bands.push_back(bi);
                    s.live -= 1;
                } else {
                    Self::queue_load(&mut s, bi, self.mb, k);
                }
                drop(s);
                self.notify();
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn fixture_pool(c: usize) -> Arc<Pool> {
        Arc::new(Pool {
            state: Mutex::new(State {
                bands: vec![Some(ActiveBand {
                    n: 0,
                    k: 16,
                    slot: None,
                    pending: c,
                    remaining: c,
                    unissued: c,
                    queued: false,
                    lifetime: 9,
                    unissued_mask: mask(c),
                    completing: false,
                    ..Default::default()
                })],
                records: (0..c).map(|_| Record { next_k: 0 }).collect(),
                slots: Vec::new(),
                stages: Vec::new(),
                free_bands: VecDeque::new(),
                free_slots: VecDeque::new(),
                free_stages: VecDeque::new(),
                loads: VecDeque::new(),
                fills: VecDeque::new(),
                ready: Bitmap::new(c),
                decoded: Bitmap::new(c),
                prev_done: Bitmap::new(c),
                next_n: 1,
                live: 1,
                pending: c,
                cursor: 0,
                lifetime: 10,
                done: false,
                error: None,
            }),
            control: ControlPort::new(1, 1),
            events: Mutex::new(VecDeque::new()),
            event_space: Semaphore::new(3),
            blocked_producers: AtomicUsize::new(0),
            completion_collectors: AtomicUsize::new(0),
            observation: Mutex::new(Observation::default()),
            manager: Notify::new(),
            filler: Notify::new(),
            issuer: Notify::new(),
            mb: c,
            capacity: 3,
            cohort: true,
            split: false,
        })
    }

    #[tokio::test]
    async fn sparse_completion_masks_publish_only_their_records_and_drain_after_last_chunk() {
        let mut a = crate::moe_normal::tests::refined_architecture(2);
        a.cores[0].refinement.as_mut().unwrap().output_pool = Some(OutputPoolConfig {
            output_contexts: 32,
            operand_stages: 2,
            scheduler_cycles: 1,
        });
        a.cores[0].refinement.as_mut().unwrap().stream_ctrl = Some(StreamControlConfig {
            event_ready: true,
            cohort_control: true,
            ..Default::default()
        });
        let core = Arc::new(CoreState::new(&a.cores[0], &a));
        let c = 17;
        let pool = fixture_pool(c);
        let finished = Arc::new(AtomicUsize::new(0));
        let ex = Executor::new();
        let (worker_pool, worker_core, worker_finished) =
            (pool.clone(), core.clone(), finished.clone());
        let clock = a.clock_period_ps;
        ex.spawn(async move {
            let even = 0x1_5555;
            let event = Event {
                kind: 2,
                source: 0,
                band: 0,
                mask: even,
                lifetime: 9,
            };
            worker_pool
                .complete_burst(&worker_core, &a, event, 16)
                .await
                .unwrap();
            {
                let s = worker_pool.state.lock().unwrap();
                assert_eq!(s.pending, 8);
                assert_eq!(s.live, 1);
                for mi in 0..17 {
                    assert_eq!(s.prev_done.get(mi), even & (1 << mi) != 0);
                }
            }
            worker_pool
                .complete_burst(
                    &worker_core,
                    &a,
                    Event {
                        mask: mask(c) ^ even,
                        ..event
                    },
                    16,
                )
                .await
                .unwrap();
            worker_finished.fetch_add(1, Ordering::SeqCst);
        });
        let (probe_pool, probe_finished) = (pool.clone(), finished.clone());
        ex.spawn(async move {
            Executor::current()
                .resolve_at(Duration::from_picos(8 * clock + clock / 2))
                .await;
            let s = probe_pool.state.lock().unwrap();
            // Pending is cleared before the full mask has been published.
            // The band must not be recycled in this interval.
            assert_eq!(s.pending, 0);
            assert_eq!(s.live, 1);
            assert!(s.bands[0].as_ref().unwrap().completing);
            probe_finished.fetch_add(1, Ordering::SeqCst);
        });
        ex.enter(Instant::ETERNITY).await;
        assert_eq!(finished.load(Ordering::SeqCst), 2);
        let s = pool.state.lock().unwrap();
        assert_eq!(s.live, 0);
        assert!(s.bands[0].is_none());
        assert!((0..c).all(|mi| s.prev_done.get(mi)));
        let r = core.report.lock().unwrap();
        let d = r.refinement.as_ref().unwrap();
        let stream = d.stream_ctrl.as_ref().unwrap();
        assert_eq!(stream.completion_mask_cycles, 6);
        assert_eq!(stream.completed_contexts, 17);
        assert_eq!(d.output_pool.as_ref().unwrap().scheduler_visits, 10);
    }
    #[tokio::test]
    async fn burst_skips_retries_once_and_retains_unissued_blocks_without_reloading() {
        let mut a = crate::moe_normal::tests::refined_architecture(2);
        a.mac_pipeline_cycles = 0;
        a.cores[0].mlen = 8;
        a.cores[0].weight_slots = 3;
        let r = a.cores[0].refinement.as_mut().unwrap();
        r.m_rows = 1;
        r.output_pool = Some(OutputPoolConfig {
            output_contexts: 32,
            operand_stages: 2,
            scheduler_cycles: 1,
        });
        r.stream_ctrl = Some(StreamControlConfig {
            event_ready: true,
            cohort_control: true,
            ..Default::default()
        });
        let core = Arc::new(CoreState::new(&a.cores[0], &a));
        let pool = fixture_pool(3);
        let p = core.config.blen;
        {
            let mut s = pool.state.lock().unwrap();
            s.pending = 0;
            let b = s.bands[0].as_mut().unwrap();
            b.k = 0;
            b.pending = 0;
            b.slot = Some(0);
            s.slots.push(Some(Slot {
                band: 0,
                tile: None,
                packed: None,
                stage: Some(0),
            }));
            s.stages.push(Stage {
                slot: Some(0),
                operands: vec![bf16::ONE; p * 8],
            });
            for mi in 0..3 {
                s.decoded.set(mi, true);
            }
            s.prev_done.set(0, true);
            s.ready.set(0, true);
        }
        let (w, _, _) = crate::moe_normal::tests::fixture();
        let region = MatrixRegion {
            rows: p,
            cols: 8,
            ..w.experts[0].gate.clone()
        };
        let finished = Arc::new(AtomicUsize::new(0));
        let ex = Executor::new();
        let (worker, worker_core, worker_finished) = (pool.clone(), core.clone(), finished.clone());
        ex.spawn(async move {
            let mut acc = vec![0.0; 3 * p];
            worker
                .cohort_issue_actor(&[bf16::ONE; 3 * 8], &mut acc, 3, &region, &worker_core, &a)
                .await
                .unwrap();
            assert!(acc[..2 * p].iter().all(|&x| x == 8.0));
            assert!(acc[2 * p..].iter().all(|&x| x == 0.0));
            worker_finished.fetch_add(1, Ordering::SeqCst);
        });
        let (probe, probe_finished) = (pool.clone(), finished.clone());
        ex.spawn(async move {
            Executor::current()
                .resolve_at(Duration::from_picos(6500))
                .await;
            {
                let mut s = probe.state.lock().unwrap();
                s.prev_done.set(1, true);
                s.ready.set(1, true);
            }
            probe.issuer.notify_one();
            Executor::current()
                .resolve_at(Duration::from_picos(100_000))
                .await;
            {
                let mut s = probe.state.lock().unwrap();
                let b = s.bands[0].as_ref().unwrap();
                assert_eq!(b.k, 0);
                assert_eq!(b.unissued_mask, 0b100);
                assert_eq!(b.slot, Some(0));
                assert!(s.loads.is_empty());
                s.done = true; // End the isolated actor fixture, not a production drain.
            }
            probe.issuer.notify_one();
            probe_finished.fetch_add(1, Ordering::SeqCst);
        });
        ex.enter(Instant::ETERNITY).await;
        assert_eq!(finished.load(Ordering::SeqCst), 2);
        let r = core.report.lock().unwrap();
        let s = r.refinement.as_ref().unwrap().stream_ctrl.as_ref().unwrap();
        assert_eq!(s.burst_starts, 1);
        assert_eq!(s.burst_partial, 1);
        assert_eq!(s.sequencer_cycles, 5);
        assert_eq!(s.burst_retry_checks, 2);
        assert_eq!(s.burst_not_ready_checks, 3);
        assert_eq!(s.writeback_order_checks, 2);
        assert_eq!(pool.events.lock().unwrap().front().unwrap().mask, 0b011);
    }
}
