//! Bounded normal-buffer output pool. Contexts outlive their weight tile and
//! operand stage, but every K tile is loaded once and served to the full M
//! cohort before its slot can be reused. No local accumulation or K reordering.
use super::*;

struct Context {
    ready: Arc<AtomicBool>,
    next_k: usize,
}

struct Band {
    n: usize,
    k: usize,
    slot: Option<usize>,
    contexts: Vec<Context>,
    pending: Arc<AtomicUsize>,
    remaining: Arc<AtomicUsize>,
    unissued: usize,
}

struct Resident {
    band: usize,
    receiver: Option<oneshot::Receiver<Result<WeightTile, String>>>,
    tile: Option<WeightTile>,
    stage: Option<usize>,
}

struct Stage {
    slot: Option<usize>,
    operands: Vec<bf16>,
}

async fn visit(core: &CoreState, a: &Architecture) {
    let cycles = core
        .config
        .refinement
        .as_ref()
        .unwrap()
        .output_pool
        .as_ref()
        .unwrap()
        .scheduler_cycles;
    let ps = (cycles * a.clock_period_ps).div_ceil(a.diagnostic.scheduler_speedup);
    Executor::current()
        .resolve_at(Duration::from_picos(ps))
        .await;
    let mut report = core.report.lock().unwrap();
    let pool = report
        .refinement
        .as_mut()
        .unwrap()
        .output_pool
        .as_mut()
        .unwrap();
    pool.scheduler_visits += 1;
    pool.scheduler_busy_ps += ps;
}

fn peaks(core: &CoreState, contexts: usize, stages: usize, pending: usize) {
    let mut report = core.report.lock().unwrap();
    let detail = report.refinement.as_mut().unwrap();
    detail.output_contexts_peak = detail.output_contexts_peak.max(contexts);
    let pool = detail.output_pool.as_mut().unwrap();
    assert!(contexts <= pool.contexts_capacity && stages <= pool.operand_stages_capacity);
    assert!(pending <= contexts);
    pool.contexts_peak = pool.contexts_peak.max(contexts);
    pool.operand_stages_peak = pool.operand_stages_peak.max(stages);
    pool.pending_contexts_peak = pool.pending_contexts_peak.max(pending);
    let mut observed = core.projection_observed.lock().unwrap();
    observed.contexts_peak = observed.contexts_peak.max(contexts);
    observed.operand_stages_peak = observed.operand_stages_peak.max(stages);
    observed.pending_contexts_peak = observed.pending_contexts_peak.max(pending);
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn gemm(
    input: &[bf16],
    output: &mut [bf16],
    accumulator: &mut [f32],
    m: usize,
    region: &MatrixRegion,
    core: &Arc<CoreState>,
    a: &Architecture,
    shared: &Arc<Shared>,
) -> Result<(), String> {
    let c = &core.config;
    let r = c.refinement.as_ref().unwrap();
    let pool = r.output_pool.as_ref().unwrap();
    let (n, k) = (region.rows, region.cols);
    let mb = m.div_ceil(r.m_rows);
    let band_capacity = (pool.output_contexts / mb).min(n.div_ceil(c.blen));
    if band_capacity == 0 {
        return Err("output pool cannot admit a complete M cohort".into());
    }
    let acc = &mut accumulator[..m * n];
    // Functional initialization only. First K injects zero; subsequent K reads
    // the old FP32 value through the finite accumulator port.
    acc.fill(0.0);
    let mut bands: Vec<Option<Band>> = (0..band_capacity).map(|_| None).collect();
    // Free IDs are links in the reserved band descriptors. Admission does not
    // scan Q occupied entries or assume a free wide priority encoder.
    let mut free_bands: VecDeque<usize> = (0..band_capacity).collect();
    let mut slots: Vec<Option<Resident>> = (0..c.weight_slots).map(|_| None).collect();
    let mut stages: Vec<Stage> = (0..pool.operand_stages)
        .map(|_| Stage {
            slot: None,
            operands: vec![bf16::ZERO; c.blen * c.mlen],
        })
        .collect();
    // At most one retirement event per admitted band. It references that
    // band's already-reserved descriptor; no unbounded result/event buffer.
    let retired: Arc<Mutex<VecDeque<usize>>> = Arc::new(Mutex::new(VecDeque::new()));
    let wake = Arc::new(Notify::new());
    let pending_total = Arc::new(AtomicUsize::new(0));
    let mut live_bands = 0;
    let mut next_n = 0;
    let mut load_cursor = 0;
    let mut issue_cursor = 0;
    let mut last_band = None;
    let ex = Executor::current();
    {
        let mut report = core.report.lock().unwrap();
        let detail = report.refinement.as_mut().unwrap();
        detail.output_context_peak_bytes = detail
            .output_context_peak_bytes
            .max(refined_context_bytes(m, c)?);
        detail.pending_result_peak_bytes = detail
            .pending_result_peak_bytes
            .max(refined_result_bytes(m, c)?);
    }
    loop {
        let mut progressed = false;
        loop {
            let index = retired.lock().unwrap().pop_front();
            let Some(index) = index else { break };
            visit(core, a).await;
            let band = bands[index].take().expect("retirement owns a live band");
            assert!(band.k >= k && band.slot.is_none());
            assert_eq!(band.pending.load(Ordering::SeqCst), 0);
            assert!(
                band.contexts
                    .iter()
                    .all(|ctx| ctx.ready.load(Ordering::SeqCst))
            );
            live_bands -= 1;
            free_bands.push_back(index);
            progressed = true;
        }
        // Whole-cohort admission; contexts never depend on obtaining a free
        // result entry during writeback. Each admission visits one descriptor.
        while next_n < n {
            let Some(index) = free_bands.pop_front() else {
                break;
            };
            visit(core, a).await;
            assert!(bands[index].is_none());
            bands[index] = Some(Band {
                n: next_n,
                k: 0,
                slot: None,
                contexts: (0..mb)
                    .map(|_| Context {
                        ready: Arc::new(AtomicBool::new(true)),
                        next_k: 0,
                    })
                    .collect(),
                pending: Arc::new(AtomicUsize::new(0)),
                remaining: Arc::new(AtomicUsize::new(mb * k.div_ceil(c.mlen))),
                unissued: mb,
            });
            next_n += c.blen;
            live_bands += 1;
            core.report
                .lock()
                .unwrap()
                .refinement
                .as_mut()
                .unwrap()
                .output_pool
                .as_mut()
                .unwrap()
                .band_admissions += 1;
            progressed = true;
        }
        if live_bands == 0 {
            assert!(next_n >= n && slots.iter().all(Option::is_none));
            assert!(stages.iter().all(|s| s.slot.is_none()));
            assert_eq!(pending_total.load(Ordering::SeqCst), 0);
            break;
        }

        // Globally allocate free weight slots to current-K demand. Allocation
        // visits bounded band descriptors in round-robin order; no future K
        // may occupy all slots while its predecessor is absent. In this first
        // pool implementation, speculative next-K prefetch is not used.
        for (slot_index, slot) in slots.iter_mut().enumerate() {
            if slot.is_some() {
                continue;
            }
            let mut chosen = None;
            for step in 0..band_capacity {
                let index = (load_cursor + step) % band_capacity;
                visit(core, a).await;
                if let Some(band) = &bands[index]
                    && band.k < k
                    && band.slot.is_none()
                    && band.pending.load(Ordering::SeqCst) < mb
                {
                    chosen = Some(index);
                    break;
                }
            }
            let Some(index) = chosen else { break };
            load_cursor = (index + 1) % band_capacity;
            let band = bands[index].as_mut().unwrap();
            let demand = shared
                .dma
                .config
                .as_ref()
                .filter(|d| d.issue_policy == ramulator::model::IssuePolicy::DemandAware)
                .map(|_| TileDemand::new(c, true));
            let rx = spawn_load_with_demand(
                core.clone(),
                shared.clone(),
                region.clone(),
                TileSpec {
                    n: band.n,
                    k: band.k,
                },
                demand,
            );
            let (tx, receiver) = oneshot::channel();
            let loaded_wake = wake.clone();
            ex.spawn(async move {
                let result = rx
                    .await
                    .unwrap_or_else(|_| Err("pool weight loader failed".into()));
                let _ = tx.send(result);
                loaded_wake.notify_one();
            });
            *slot = Some(Resident {
                band: index,
                receiver: Some(receiver),
                tile: None,
                stage: None,
            });
            band.slot = Some(slot_index);
            core.report
                .lock()
                .unwrap()
                .refinement
                .as_mut()
                .unwrap()
                .output_pool
                .as_mut()
                .unwrap()
                .tile_admissions += 1;
            progressed = true;
        }
        // Poll only resident loading descriptors, each at a charged scheduler
        // visit. Notifications retain a permit across these timed checks.
        for resident in slots.iter_mut().flatten() {
            if let Some(rx) = &mut resident.receiver {
                visit(core, a).await;
                match rx.try_recv() {
                    Ok(value) => {
                        resident.tile = Some(value?);
                        resident.receiver = None;
                        progressed = true;
                    }
                    Err(oneshot::error::TryRecvError::Empty) => (),
                    Err(oneshot::error::TryRecvError::Closed) => {
                        return Err("pool tile completion channel closed".into());
                    }
                }
            }
        }
        for (stage_index, stage) in stages.iter_mut().enumerate() {
            if stage.slot.is_some() {
                continue;
            }
            for (slot_index, entry) in slots.iter_mut().enumerate() {
                visit(core, a).await;
                let Some(resident) = entry else { continue };
                if resident.stage.is_some() || resident.tile.is_none() {
                    continue;
                }
                refined_port_work(core, true, c.blen * c.mlen, a.clock_period_ps).await;
                stage
                    .operands
                    .copy_from_slice(&resident.tile.as_ref().unwrap().values);
                stage.slot = Some(slot_index);
                resident.stage = Some(stage_index);
                progressed = true;
                break;
            }
        }
        peaks(
            core,
            live_bands * mb,
            stages.iter().filter(|s| s.slot.is_some()).count(),
            pending_total.load(Ordering::SeqCst),
        );

        // Round robin across stages, then ascending M within its resident tile.
        // Every checked context costs a scheduler visit. The context cannot
        // issue another K until its previous FP32 writeback has completed.
        let mut selected = None;
        'select: for step in 0..stages.len() {
            let si = (issue_cursor + step) % stages.len();
            let Some(slot_index) = stages[si].slot else {
                continue;
            };
            let resident = slots[slot_index].as_ref().unwrap();
            let band = bands[resident.band].as_ref().unwrap();
            for (mi, context) in band.contexts.iter().enumerate() {
                visit(core, a).await;
                if context.next_k == band.k && context.ready.load(Ordering::SeqCst) {
                    selected = Some((si, slot_index, resident.band, mi));
                    break 'select;
                }
            }
        }
        let Some((si, slot_index, bi, mi)) = selected else {
            if progressed {
                continue;
            }
            let waiting_feedback = bands
                .iter()
                .flatten()
                .any(|b| b.pending.load(Ordering::SeqCst) > 0)
                && (slots.iter().flatten().any(|s| s.tile.is_some())
                    || slots.iter().flatten().all(|s| s.receiver.is_none()));
            let final_drain = next_n >= n && bands.iter().flatten().all(|b| b.k >= k);
            let begin = ex.now().as_picos();
            wake.notified().await;
            let elapsed = ex.now().as_picos() - begin;
            let mut report = core.report.lock().unwrap();
            if final_drain {
                report.pipeline_drain_ps += elapsed;
            } else if waiting_feedback {
                report.accumulator_dependency_stall_ps += elapsed;
                report.refinement.as_mut().unwrap().output_context_stall_ps += elapsed;
            } else {
                report.weight_ready_wait_ps += elapsed;
            }
            continue;
        };
        let band = bands[bi].as_mut().unwrap();
        let m0 = mi * r.m_rows;
        let mr = r.m_rows.min(m - m0);
        let nr = c.blen.min(n - band.n);
        let kr = c.mlen.min(k - band.k);
        if band.k > 0 {
            refined_port_work(core, false, mr * nr, a.clock_period_ps).await;
        }
        for row in 0..mr {
            for col in 0..nr {
                let target = (m0 + row) * n + band.n + col;
                for kk in 0..kr {
                    let product = input[(m0 + row) * k + band.k + kk].to_f32()
                        * stages[si].operands[col * c.mlen + kk].to_f32();
                    acc[target] += product;
                }
            }
        }
        let issue_rows = match r.tail_policy {
            TailPolicy::Padded => r.m_rows,
            TailPolicy::ValidRows => mr,
        };
        let service = issue_service_ps(a, c, issue_rows);
        let result_time = ex.now() + Duration::from_picos(service + feedback_ps(a, c));
        let context = &mut band.contexts[mi];
        assert!(context.ready.swap(false, Ordering::SeqCst));
        context.next_k += c.mlen;
        band.unissued -= 1;
        band.pending.fetch_add(1, Ordering::SeqCst);
        let pending = pending_total.fetch_add(1, Ordering::SeqCst) + 1;
        let done = context.ready.clone();
        let band_pending = band.pending.clone();
        let band_remaining = band.remaining.clone();
        let completion_pending = pending_total.clone();
        let completion_core = core.clone();
        let completion_wake = wake.clone();
        let completion_retired = retired.clone();
        let clock = a.clock_period_ps;
        ex.spawn(async move {
            Executor::current().resolve_at(result_time).await;
            refined_port_work(&completion_core, false, mr * nr, clock).await;
            done.store(true, Ordering::SeqCst);
            band_pending.fetch_sub(1, Ordering::SeqCst);
            completion_pending.fetch_sub(1, Ordering::SeqCst);
            if band_remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                let mut queue = completion_retired.lock().unwrap();
                assert!(queue.len() < band_capacity);
                queue.push_back(bi);
            }
            completion_wake.notify_one();
        });
        {
            let mut report = core.report.lock().unwrap();
            report.useful_macs += (mr * nr * kr) as u64;
            report.issued_macs += (issue_rows * c.blen * c.mlen) as u64;
            report.compute_busy_ps += service;
            let detail = report.refinement.as_mut().unwrap();
            if last_band.is_some_and(|prev| prev != bi) {
                detail.independent_output_switches += 1;
            }
            detail.output_pool.as_mut().unwrap().context_updates += 1;
        }
        peaks(
            core,
            live_bands * mb,
            stages.iter().filter(|s| s.slot.is_some()).count(),
            pending,
        );
        last_band = Some(bi);
        issue_cursor = (si + 1) % stages.len();
        ex.resolve_at(Duration::from_picos(service)).await;
        if band.unissued == 0 {
            // The tile has served every M consumer. Only this point releases
            // its operand latch and packed/decoded slot; FP32 feedback remains
            // owned by the Q records until the independent port drain finishes.
            slots[slot_index].take();
            stages[si].slot = None;
            band.slot = None;
            band.k += c.mlen;
            band.unissued = mb;
        }
    }
    // No speculative loaders or pending writebacks survive projection end.
    let finalize_start = ex.now().as_picos();
    refined_port_work(core, false, m * n, a.clock_period_ps).await;
    let vector_wait = shared.vector.work(m * n).await;
    {
        let mut report = core.report.lock().unwrap();
        report.vector_wait_ps += vector_wait;
        let detail = report.refinement.as_mut().unwrap();
        detail.finalized_elements += (m * n) as u64;
        detail.output_finalize_elapsed_ps += ex.now().as_picos() - finalize_start;
    }
    for (value, sum) in output.iter_mut().zip(acc.iter().copied()) {
        *value = bf16::from_f32(sum);
        if !value.is_finite() {
            return Err("GEMM produced non-finite BF16".into());
        }
    }
    Ok(())
}
