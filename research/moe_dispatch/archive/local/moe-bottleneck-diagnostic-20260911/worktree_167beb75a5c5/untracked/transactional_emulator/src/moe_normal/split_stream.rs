//! Split packed/operand lifetimes on the same cohort issue and completion path.
use super::*;

impl Pool {
    /// Candidate headers are finite registers (16 B * W), populated through
    /// the same descriptor port before the comparator may observe them.
    pub(super) async fn refill_window(&self, core: &CoreState, a: &Architecture) -> bool {
        if !self.split {
            return false;
        }
        let next = {
            let s = self.state.lock().unwrap();
            s.loads
                .iter()
                .take(self.capacity)
                .copied()
                .find(|&bi| s.bands[bi].as_ref().unwrap().window_entered_ps.is_none())
        };
        let Some(bi) = next else {
            return false;
        };
        let _port = self
            .access(core, a, 1, false, None, bi, ControlCostClass::Header)
            .await;
        let mut s = self.state.lock().unwrap();
        s.bands[bi].as_mut().unwrap().window_entered_ps =
            Some(Executor::current().now().as_picos());
        core.split_observe(|r| r.window_header_updates += 1);
        true
    }

    pub(super) fn observe_full_waits(&self) {
        if !self.split {
            return;
        }
        let now = Executor::current().now().as_picos();
        let s = self.state.lock().unwrap();
        let landing = !s.loads.is_empty() && s.free_slots.is_empty();
        let stage = !s.fills.is_empty() && s.free_stages.is_empty();
        let mut o = self.observation.lock().unwrap();
        if landing {
            o.landing_full_since.get_or_insert(now);
        } else if let Some(start) = o.landing_full_since.take() {
            o.landing_full_ps += now - start;
        }
        if stage {
            o.stage_full_since.get_or_insert(now);
        } else if let Some(start) = o.stage_full_since.take() {
            o.stage_full_ps += now - start;
        }
    }

    pub(super) fn select_window(&self, s: &mut State, core: &CoreState) -> Option<usize> {
        let now = Executor::current().now().as_picos();
        let threshold = core.config.split_window().unwrap().threshold_ps().unwrap();
        let candidates: Vec<_> = s
            .loads
            .iter()
            .take(self.capacity)
            .enumerate()
            .map(|(i, &bi)| (i, bi))
            .collect();
        let ready: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|&(_, bi)| {
                let b = s.bands[bi].as_ref().unwrap();
                b.window_entered_ps.is_some() && b.slot.is_none() && b.operand_stage.is_none()
            })
            .collect();
        let normal = ready
            .iter()
            .min_by_key(|&&(_, bi)| {
                let b = s.bands[bi].as_ref().unwrap();
                (b.n, b.k)
            })
            .copied();
        let aged = threshold.and_then(|limit| {
            ready
                .iter()
                .copied()
                .filter(|&(_, bi)| {
                    now - s.bands[bi]
                        .as_ref()
                        .unwrap()
                        .window_entered_ps
                        .unwrap_or(now)
                        >= limit
                })
                .min_by_key(|&(_, bi)| {
                    let b = s.bands[bi].as_ref().unwrap();
                    (b.window_entered_ps.unwrap_or(now), b.n, b.k)
                })
        });
        let selected = aged.or(normal);
        core.split_observe(|r| {
            r.window_candidate_checks += candidates.len() as u64;
            r.window_max_candidates = r.window_max_candidates.max(candidates.len());
            r.window_selections += u64::from(selected.is_some());
            r.aged_selected += u64::from(aged.is_some());
            r.aging_reorders += u64::from(aged.is_some() && aged != normal);
        });
        selected.map(|(i, _)| i)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_packed(
        self: &Arc<Self>,
        core: &Arc<CoreState>,
        shared: &Arc<Shared>,
        region: &MatrixRegion,
        tile: TileSpec,
        si: usize,
        bi: usize,
        lifetime: u64,
    ) {
        let receiver = spawn_packed_load(core.clone(), shared.clone(), region.clone(), tile);
        let (pool, core) = (self.clone(), core.clone());
        Executor::current().spawn(async move {
            match receiver
                .await
                .unwrap_or_else(|_| Err("split packed loader failed".into()))
            {
                Ok(tile) => {
                    {
                        let mut s = pool.state.lock().unwrap();
                        assert_eq!(s.bands[bi].as_ref().unwrap().lifetime, lifetime);
                        s.slots[si].as_mut().unwrap().packed = Some(tile);
                    }
                    pool.send(
                        &core,
                        Event {
                            kind: 0,
                            source: si as u8,
                            band: bi as u16,
                            mask: 0,
                            lifetime,
                        },
                    )
                    .await;
                }
                Err(error) => {
                    pool.state.lock().unwrap().error = Some(error);
                    pool.notify();
                }
            }
        });
    }

    pub(super) async fn split_operand_actor(
        self: &Arc<Self>,
        core: &Arc<CoreState>,
        a: &Architecture,
        shared: &Arc<Shared>,
        _region: &MatrixRegion,
    ) -> Result<(), String> {
        let c = &core.config;
        let ex = Executor::current();
        let elements = c.blen * c.mlen;
        let stage_bytes = (elements * 2) as i64;
        let width = c
            .refinement
            .as_ref()
            .unwrap()
            .weight_read_elements_per_cycle;
        loop {
            self.observe_full_waits();
            let notification = self.filler.notified();
            tokio::pin!(notification);
            notification.as_mut().enable();
            let work = {
                let mut s = self.state.lock().unwrap();
                if s.done {
                    return Ok(());
                }
                if !s.free_stages.is_empty() && !s.fills.is_empty() {
                    Some((
                        s.free_stages.pop_front().unwrap(),
                        s.fills.pop_front().unwrap(),
                    ))
                } else {
                    None
                }
            };
            let Some((stage, slot)) = work else {
                notification.await;
                continue;
            };
            core.change_lifetime(0, 0, stage_bytes);
            self.observe_full_waits();
            let bi = self.state.lock().unwrap().slots[slot]
                .as_ref()
                .unwrap()
                .band;
            let (packed, lifetime) = {
                let _port = self
                    .access(core, a, 2, false, None, bi, ControlCostClass::OperandBind)
                    .await;
                let mut s = self.state.lock().unwrap();
                let packed = s.slots[slot]
                    .as_mut()
                    .unwrap()
                    .packed
                    .take()
                    .expect("both packed streams have arrived");
                s.stages[stage].slot = Some(bi); // split mode: stage stores band owner, not a reusable packed slot ID
                let b = s.bands[bi].as_mut().unwrap();
                assert!(b.operand_stage.is_none());
                b.operand_stage = Some(stage);
                (packed, b.lifetime)
            };
            self.update_peaks(core);
            let begin = ex.now().as_picos();
            core.split_observe(|r| r.decode_queue_wait_ps += begin - packed.arrived_ps);
            // Existing port units are BF16 elements; packed read includes scales.
            let packed_elements = packed.bytes.len().div_ceil(2);
            let read_busy = (packed_elements.div_ceil(width) as u64 * a.clock_period_ps)
                .div_ceil(a.diagnostic.weight_port_speedup);
            refined_port_work(core, true, packed_elements, a.clock_period_ps).await;
            core.split_observe(|r| {
                r.packed_read_busy_ps += read_busy;
                r.packed_read_wait_ps += ex.now().as_picos() - begin - read_busy;
            });
            let vector_start = ex.now().as_picos();
            let vector_wait = shared.vector.work(packed.nr * packed.kr).await;
            core.report.lock().unwrap().vector_wait_ps += vector_wait;
            core.split_observe(|r| {
                r.decode_vector_wait_ps += vector_wait;
                r.decode_vector_busy_ps += ex.now().as_picos() - vector_start - vector_wait;
            });
            let write_start = ex.now().as_picos();
            let write_busy = (elements.div_ceil(width) as u64 * a.clock_period_ps)
                .div_ceil(a.diagnostic.weight_port_speedup);
            refined_port_work(core, true, elements, a.clock_period_ps).await;
            core.split_observe(|r| {
                r.operand_write_busy_ps += write_busy;
                r.operand_write_wait_ps += ex.now().as_picos() - write_start - write_busy;
            });
            {
                let mut s = self.state.lock().unwrap();
                packed.decode_into(&mut s.stages[stage].operands, c)?;
            }
            core.change_lifetime(0, stage_bytes, -stage_bytes);
            let load_elapsed = ex.now().as_picos() - packed.issued_ps;
            record_load(&mut core.report.lock().unwrap().tile_loads, load_elapsed);
            record_load(
                &mut core.projection_observed.lock().unwrap().tile_loads,
                load_elapsed,
            );
            {
                // Packed-slot and band ownership writes. Operand retirement
                // later pays its own two cycles: total tile overhead is 9T.
                let _port = self
                    .access(core, a, 2, false, None, bi, ControlCostClass::PackedRetire)
                    .await;
                let mut s = self.state.lock().unwrap();
                assert_eq!(s.bands[bi].as_ref().unwrap().slot, Some(slot));
                s.slots[slot].take();
                s.free_slots.push_back(slot);
                s.bands[bi].as_mut().unwrap().slot = None;
            }
            drop(packed); // Actual packed bytes and reservation end together.
            {
                let mut report = core.report.lock().unwrap();
                let r = report
                    .refinement
                    .as_mut()
                    .unwrap()
                    .stream_ctrl
                    .as_mut()
                    .unwrap();
                r.operand_fill_elapsed_ps += ex.now().as_picos() - begin;
                r.operand_fill_busy_ps += read_busy + write_busy;
            }
            self.notify();
            self.send(
                core,
                Event {
                    kind: 1,
                    source: stage as u8,
                    band: bi as u16,
                    mask: 0,
                    lifetime,
                },
            )
            .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn aged_head_bypasses_blocked_entries_without_peeking_outside_paid_window() {
        let mut a = crate::moe_normal::tests::refined_architecture(2);
        let r = a.cores[0].refinement.as_mut().unwrap();
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
                window_tiles: 3,
                load_latency_sum_ps: 100,
                load_latency_samples: 1,
                aging_multiplier: Some(4),
            }),
            ..Default::default()
        });
        let core = Arc::new(CoreState::new(&a.cores[0], &a));
        a.cores[0]
            .refinement
            .as_mut()
            .unwrap()
            .stream_ctrl
            .as_mut()
            .unwrap()
            .split_window
            .as_mut()
            .unwrap()
            .aging_multiplier = None;
        let unaged = Arc::new(CoreState::new(&a.cores[0], &a));
        let mut pool = super::super::cohort::tests::fixture_pool(1);
        Arc::get_mut(&mut pool).unwrap().split = true;
        {
            let mut s = pool.state.lock().unwrap();
            s.bands = vec![
                Some(ActiveBand {
                    n: 0,
                    slot: Some(0),
                    window_entered_ps: Some(0),
                    ..Default::default()
                }),
                Some(ActiveBand {
                    n: 1,
                    window_entered_ps: Some(900),
                    ..Default::default()
                }),
                Some(ActiveBand {
                    n: 2,
                    window_entered_ps: Some(0),
                    ..Default::default()
                }),
                Some(ActiveBand {
                    n: 0,
                    window_entered_ps: Some(0),
                    ..Default::default()
                }),
            ];
            s.loads = VecDeque::from([0, 1, 2, 3]);
        }
        let ex = Executor::new();
        ex.spawn(async move {
            Executor::current()
                .resolve_at(Duration::from_picos(1000))
                .await;
            let mut s = pool.state.lock().unwrap();
            assert_eq!(pool.select_window(&mut s, &core), Some(2));
            assert_eq!(pool.select_window(&mut s, &unaged), Some(1));
            let report = core.report.lock().unwrap();
            let r = report
                .refinement
                .as_ref()
                .unwrap()
                .split_window
                .as_ref()
                .unwrap();
            assert_eq!((r.window_max_candidates, r.aging_reorders), (3, 1));
            s.bands[1].as_mut().unwrap().operand_stage = Some(0);
            s.bands[2].as_mut().unwrap().operand_stage = Some(1);
            assert_eq!(pool.select_window(&mut s, &unaged), None); // index 3 is eligible but outside W
        });
        ex.enter(Instant::ETERNITY).await;
    }
}
