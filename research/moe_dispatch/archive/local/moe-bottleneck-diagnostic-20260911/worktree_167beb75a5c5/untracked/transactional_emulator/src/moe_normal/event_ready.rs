//! Event-driven output pool with unchanged data buffers and arithmetic.
//! Three actors share a single, charged control port. Slot/context completion
//! producers are bounded by S + Q; a full W-entry FIFO backpressures them.
use super::*;
#[path = "cohort.rs"]
mod cohort;
#[path = "split_stream.rs"]
mod split_stream;

#[derive(Clone)]
struct Bitmap {
    words: Vec<u64>,
    len: usize,
}
impl Bitmap {
    fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }
    fn get(&self, i: usize) -> bool {
        self.words[i / 64] & (1 << (i % 64)) != 0
    }
    fn set(&mut self, i: usize, value: bool) {
        assert!(i < self.len);
        let mask = 1 << (i % 64);
        if value {
            self.words[i / 64] |= mask;
        } else {
            self.words[i / 64] &= !mask;
        }
    }
    fn first_from(&self, begin: usize) -> Option<usize> {
        if begin >= self.len {
            return None;
        }
        let start = begin / 64;
        for wi in start..self.words.len() {
            let word = self.words[wi]
                & if wi == start {
                    u64::MAX << (begin % 64)
                } else {
                    u64::MAX
                };
            if word != 0 {
                return Some(wi * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }
    fn select(&self, cursor: usize, mode: ReadySelection) -> Option<usize> {
        // At most four words (Q<=256), modelling a one-cycle bounded encoder,
        // not a zero-cost walk through Q descriptors.
        match mode {
            ReadySelection::Lowest => self.first_from(0),
            ReadySelection::Rotating | ReadySelection::BandRotating => {
                self.first_from(cursor).or_else(|| self.first_from(0))
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Event {
    kind: u8, // 0: slot arrived (packed in split mode), 1: operands decoded/installed, 2: record/burst K done
    source: u8,
    band: u16,
    mask: u32, // relative M-cohort bits; zero on the frozen record path
    lifetime: u64,
}

struct Record {
    next_k: usize,
}
#[derive(Default)]
struct ActiveBand {
    n: usize,
    k: usize,
    slot: Option<usize>,
    pending: usize,
    remaining: usize,
    unissued: usize,
    queued: bool,
    lifetime: u64,
    /// Replaces per-M descriptor K writes in cohort mode; within the already
    /// paid 64 B band record. The sequencer updates it without a control visit.
    unissued_mask: u32,
    completing: bool, // prevent retirement during multi-cycle mask publication
    operand_stage: Option<usize>, // split stage survives packed-slot release
    prefetch_pending: bool,
    queued_ps: u64,
    window_entered_ps: Option<u64>,
}
#[derive(Default)]
struct Slot {
    band: usize,
    tile: Option<WeightTile>,
    packed: Option<PackedTile>,
    stage: Option<usize>,
}
struct State {
    bands: Vec<Option<ActiveBand>>,
    records: Vec<Record>,
    slots: Vec<Option<Slot>>,
    stages: Vec<Stage>,
    // Physical implementations use links in the already charged band/slot/
    // stage descriptors. These host queues add no architectural entries.
    free_bands: VecDeque<usize>,
    free_slots: VecDeque<usize>,
    free_stages: VecDeque<usize>,
    loads: VecDeque<usize>,
    fills: VecDeque<usize>,
    ready: Bitmap,
    decoded: Bitmap,
    prev_done: Bitmap,
    next_n: usize,
    live: usize,
    pending: usize,
    cursor: usize,
    lifetime: u64,
    done: bool,
    error: Option<String>,
}

/// Simulation telemetry only: never read by readiness or resource decisions.
/// Queue timestamps mirror the bounded event FIFO; intervals are reduced at
/// projection drain and do not represent an extra hardware buffer.
#[derive(Default)]
struct Observation {
    queued: VecDeque<(u64, u64)>, // producer-ready time, FIFO insertion time
    event_service: Vec<(u64, u64)>,
    mac_service: Vec<(u64, u64)>,
    landing_full_since: Option<u64>,
    landing_full_ps: u64,
    stage_full_since: Option<u64>,
    stage_full_ps: u64,
}

/// Both inputs are chronological, internally disjoint half-open intervals.
fn overlap_ps(left: &[(u64, u64)], right: &[(u64, u64)]) -> u64 {
    let (mut i, mut j, mut sum) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        let (a, b) = (left[i], right[j]);
        sum += a.1.min(b.1).saturating_sub(a.0.max(b.0));
        if a.1 <= b.1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    sum
}

struct Pool {
    state: Mutex<State>,
    control: ControlPort,
    events: Mutex<VecDeque<Event>>,
    event_space: Semaphore,
    blocked_producers: AtomicUsize, // observer, never consulted for decisions
    completion_collectors: AtomicUsize, // observer of bounded pending bursts
    observation: Mutex<Observation>,
    manager: Notify,
    filler: Notify,
    issuer: Notify,
    mb: usize,
    capacity: usize,
    cohort: bool,
    split: bool,
}

impl Pool {
    fn notify(&self) {
        self.observe_full_waits();
        self.manager.notify_one();
        self.filler.notify_one();
        self.issuer.notify_one();
    }

    /// One descriptor RMW per cycle. Selection may accompany that access,
    /// exactly as the old scheduler's charged visit; fan-out is never free.
    #[allow(clippy::too_many_arguments)]
    async fn access(
        &self,
        core: &CoreState,
        a: &Architecture,
        writes: u64,
        selector: bool,
        event_kind: Option<usize>,
        band: usize,
        class: ControlCostClass,
    ) -> ControlLease<'_> {
        let cycles = writes.max(1);
        let duration = if a.diagnostic.oracle_control.removes(class)
            || (event_kind.is_some() && a.diagnostic.event_updates_zero_cost)
        {
            0 // Explicit diagnostic oracle; never used for acceptance.
        } else {
            (cycles * a.clock_period_ps).div_ceil(a.diagnostic.scheduler_speedup)
        };
        let permit = self.control.access(duration, band).await;
        let service_start = permit.start;
        if event_kind.is_some() {
            self.observation
                .lock()
                .unwrap()
                .event_service
                .push((service_start, service_start + duration));
        }
        let mut report = core.report.lock().unwrap();
        let detail = report.refinement.as_mut().unwrap();
        let pool = detail.output_pool.as_mut().unwrap();
        pool.scheduler_visits += cycles;
        pool.scheduler_busy_ps += duration;
        let stream = detail.stream_ctrl.as_mut().unwrap();
        let oracle = stream
            .oracle_control
            .get_or_insert_with(|| OracleControlReport {
                mode: a.diagnostic.oracle_control,
                ..Default::default()
            });
        let observed = oracle.categories.entry(class).or_default();
        observed.accesses += 1;
        observed.nominal_cycles += cycles;
        observed.paid_ps += duration;
        observed.wait_ps += permit.wait;
        stream.descriptor_reads += cycles;
        stream.descriptor_writes += writes;
        stream.scheduler_port_wait_ps += permit.wait;
        stream.control_conflict_wait_ps += permit.conflict_wait;
        if selector {
            stream.selector_visits += 1;
            stream.selector_busy_ps += duration;
        }
        if let Some(kind) = event_kind {
            stream.event_update_cycles_by_kind[kind] += cycles;
            stream.event_service_ps_by_kind[kind] += duration;
            stream.event_control_wait_ps_by_kind[kind] += permit.wait;
        }
        if event_kind == Some(1) {
            stream.fanout_writes += writes;
            stream.event_fanout_busy_ps += duration;
        }
        permit
    }

    async fn send(&self, core: &CoreState, event: Event) {
        let ex = Executor::current();
        let start = ex.now().as_picos();
        let blocked = self.event_space.available_permits() == 0;
        if blocked {
            let count = self.blocked_producers.fetch_add(1, Ordering::SeqCst) + 1;
            let mut report = core.report.lock().unwrap();
            let stream = report
                .refinement
                .as_mut()
                .unwrap()
                .stream_ctrl
                .as_mut()
                .unwrap();
            stream.blocked_producers_peak = stream.blocked_producers_peak.max(count);
        }
        let permit = self.event_space.acquire().await.unwrap();
        permit.forget();
        if blocked {
            self.blocked_producers.fetch_sub(1, Ordering::SeqCst);
        }
        let count = {
            let mut events = self.events.lock().unwrap();
            assert!(events.len() < self.capacity);
            events.push_back(event);
            events.len()
        };
        self.observation
            .lock()
            .unwrap()
            .queued
            .push_back((start, ex.now().as_picos()));
        {
            let mut report = core.report.lock().unwrap();
            let stream = report
                .refinement
                .as_mut()
                .unwrap()
                .stream_ctrl
                .as_mut()
                .unwrap();
            stream.events_enqueued += 1;
            stream.event_queue_peak = stream.event_queue_peak.max(count);
            stream.event_queue_wait_ps += ex.now().as_picos() - start;
        }
        self.manager.notify_one();
    }

    fn queue_load(s: &mut State, bi: usize, mb: usize, k: usize) {
        let b = s.bands[bi].as_mut().unwrap();
        if b.slot.is_none()
            && b.operand_stage.is_none()
            && b.k < k
            && (b.pending < mb || b.prefetch_pending)
            && !b.queued
        {
            b.queued = true;
            if b.prefetch_pending {
                b.queued_ps = Executor::current().now().as_picos();
                b.window_entered_ps = None;
            }
            assert!(s.loads.len() < s.bands.len());
            s.loads.push_back(bi);
        }
    }

    fn update_peaks(&self, core: &CoreState) {
        let s = self.state.lock().unwrap();
        peaks(
            core,
            s.live * self.mb,
            s.stages.iter().filter(|s| s.slot.is_some()).count(),
            s.pending,
        );
    }

    async fn events_actor(
        self: &Arc<Self>,
        core: &Arc<CoreState>,
        a: &Architecture,
        shared: &Arc<Shared>,
        region: &MatrixRegion,
    ) -> Result<(), String> {
        let c = &core.config;
        let (n, k) = (region.rows, region.cols);
        let ex = Executor::current();
        loop {
            self.observe_full_waits();
            let notification = self.manager.notified();
            tokio::pin!(notification);
            notification.as_mut().enable();
            if let Some(error) = self.state.lock().unwrap().error.clone() {
                return Err(error);
            }
            let event = self.events.lock().unwrap().pop_front();
            if let Some(event) = event {
                self.event_space.add_permits(1);
                let (producer_ready, enqueued) = self
                    .observation
                    .lock()
                    .unwrap()
                    .queued
                    .pop_front()
                    .expect("one timestamp per enqueued event");
                let event_start = ex.now().as_picos();
                let bi = event.band as usize;
                {
                    let s = self.state.lock().unwrap();
                    assert_eq!(s.bands[bi].as_ref().unwrap().lifetime, event.lifetime);
                }
                match event.kind {
                    0 => {
                        let _port = self
                            .access(core, a, 1, false, Some(0), bi, ControlCostClass::Arrival)
                            .await;
                        let mut s = self.state.lock().unwrap();
                        assert!(s.fills.len() < c.resident_slots());
                        s.fills.push_back(event.source as usize);
                    }
                    1 => {
                        // One context descriptor write each cycle. Other actors
                        // can consume early contexts while this fan-out proceeds.
                        for mi in 0..self.mb {
                            let _port = self
                                .access(core, a, 1, false, Some(1), bi, ControlCostClass::Install)
                                .await;
                            let mut s = self.state.lock().unwrap();
                            let ci = bi * self.mb + mi;
                            s.decoded.set(ci, true);
                            let b = s.bands[bi].as_ref().unwrap();
                            let not_issued = if self.cohort {
                                b.unissued_mask & (1 << mi) != 0
                            } else {
                                s.records[ci].next_k == b.k
                            };
                            let ready = s.prev_done.get(ci) && not_issued;
                            s.ready.set(ci, ready);
                            drop(s);
                            self.issuer.notify_one();
                        }
                    }
                    2 if self.cohort => self.complete_burst(core, a, event, k).await?,
                    2 => {
                        // Context and band completion counters are separate
                        // records, hence two charged descriptor writes.
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
                        let ci = event.source as usize;
                        s.prev_done.set(ci, true);
                        let ready = s.decoded.get(ci)
                            && s.records[ci].next_k == s.bands[bi].as_ref().unwrap().k;
                        s.ready.set(ci, ready);
                        s.pending -= 1;
                        let b = s.bands[bi].as_mut().unwrap();
                        b.pending -= 1;
                        b.remaining -= 1;
                        if b.remaining == 0 && b.slot.is_none() && b.operand_stage.is_none() {
                            assert!(b.slot.is_none() && b.k >= k);
                            s.bands[bi] = None;
                            s.free_bands.push_back(bi);
                            s.live -= 1;
                        } else {
                            Self::queue_load(&mut s, bi, self.mb, k);
                        }
                    }
                    _ => return Err("invalid output-pool event".into()),
                }
                {
                    let mut report = core.report.lock().unwrap();
                    let stream = report
                        .refinement
                        .as_mut()
                        .unwrap()
                        .stream_ctrl
                        .as_mut()
                        .unwrap();
                    let kind = event.kind as usize;
                    let delivery = ex.now().as_picos() - producer_ready;
                    stream.events_processed += 1;
                    stream.event_counts_by_kind[kind] += 1;
                    stream.event_queue_residence_ps_by_kind[kind] += event_start - enqueued;
                    stream.event_delivery_ps_by_kind[kind] += delivery;
                    stream.event_delivery_max_ps_by_kind[kind] =
                        stream.event_delivery_max_ps_by_kind[kind].max(delivery);
                }
                self.update_peaks(core);
                self.notify();
                continue;
            }
            let admission = {
                let mut s = self.state.lock().unwrap();
                if s.next_n < n {
                    s.free_bands.pop_front()
                } else {
                    None
                }
            };
            if let Some(bi) = admission {
                let _port = self
                    .access(
                        core,
                        a,
                        1 + self.mb as u64,
                        false,
                        None,
                        bi,
                        ControlCostClass::Admission,
                    )
                    .await;
                let mut s = self.state.lock().unwrap();
                let lifetime = s.lifetime;
                s.lifetime = s.lifetime.checked_add(1).ok_or("band lifetime overflow")?;
                s.bands[bi] = Some(ActiveBand {
                    n: s.next_n,
                    k: 0,
                    slot: None,
                    pending: 0,
                    remaining: self.mb * k.div_ceil(c.mlen),
                    unissued: self.mb,
                    queued: false,
                    lifetime,
                    unissued_mask: if self.cohort {
                        cohort::mask(self.mb)
                    } else {
                        0
                    },
                    completing: false,
                    prefetch_pending: self.split,
                    ..Default::default()
                });
                for ci in bi * self.mb..(bi + 1) * self.mb {
                    s.records[ci].next_k = 0;
                    s.prev_done.set(ci, true);
                    s.decoded.set(ci, false);
                    s.ready.set(ci, false);
                }
                s.next_n += c.blen;
                s.live += 1;
                Self::queue_load(&mut s, bi, self.mb, k);
                drop(s);
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
                self.update_peaks(core);
                continue;
            }
            if self.refill_window(core, a).await {
                continue;
            }
            let load = {
                let mut s = self.state.lock().unwrap();
                if !s.free_slots.is_empty() && !s.loads.is_empty() {
                    let index = if self.split {
                        self.select_window(&mut s, core)
                    } else {
                        Some(0)
                    };
                    index.map(|i| {
                        (
                            s.free_slots.pop_front().unwrap(),
                            s.loads.remove(i).unwrap(),
                        )
                    })
                } else {
                    None
                }
            };
            if let Some((si, bi)) = load {
                self.observe_full_waits();
                let _port = self
                    .access(core, a, 2, false, None, bi, ControlCostClass::Load)
                    .await;
                let (tile, lifetime) = {
                    let mut s = self.state.lock().unwrap();
                    let b = s.bands[bi].as_mut().unwrap();
                    b.queued = false;
                    b.slot = Some(si);
                    if self.split {
                        core.split_observe(|r| {
                            r.window_wait_ps += ex.now().as_picos() - b.queued_ps
                        });
                    }
                    let out = (TileSpec { n: b.n, k: b.k }, b.lifetime);
                    s.slots[si] = Some(Slot {
                        band: bi,
                        tile: None,
                        packed: None,
                        stage: None,
                    });
                    out
                };
                if self.split {
                    self.launch_packed(core, shared, region, tile, si, bi, lifetime);
                } else {
                    let receiver = spawn_load(core.clone(), shared.clone(), region.clone(), tile);
                    let pool = self.clone();
                    let loaded_core = core.clone();
                    ex.spawn(async move {
                        match receiver
                            .await
                            .unwrap_or_else(|_| Err("event pool loader failed".into()))
                        {
                            Ok(tile) => {
                                pool.state.lock().unwrap().slots[si].as_mut().unwrap().tile =
                                    Some(tile);
                                pool.send(
                                    &loaded_core,
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
                continue;
            }
            {
                let mut s = self.state.lock().unwrap();
                if s.next_n >= n && s.live == 0 {
                    assert_eq!(s.pending, 0);
                    assert!(s.slots.iter().all(Option::is_none));
                    assert!(s.stages.iter().all(|x| x.slot.is_none()));
                    assert_eq!(self.event_space.available_permits(), self.capacity);
                    s.done = true;
                    drop(s);
                    self.notify();
                    return Ok(());
                }
            }
            notification.await;
        }
    }

    async fn operand_actor(
        self: &Arc<Self>,
        core: &Arc<CoreState>,
        a: &Architecture,
        shared: &Arc<Shared>,
        region: &MatrixRegion,
    ) -> Result<(), String> {
        let c = &core.config;
        if self.split {
            return self.split_operand_actor(core, a, shared, region).await;
        }
        loop {
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
            let (bi, lifetime) = {
                let bi = self.state.lock().unwrap().slots[slot]
                    .as_ref()
                    .unwrap()
                    .band;
                let _port = self
                    .access(core, a, 2, false, None, bi, ControlCostClass::OperandBind)
                    .await;
                let mut s = self.state.lock().unwrap();
                let resident = s.slots[slot].as_mut().unwrap();
                resident.stage = Some(stage);
                let bi = resident.band;
                s.stages[stage].slot = Some(slot);
                (bi, s.bands[bi].as_ref().unwrap().lifetime)
            };
            let start = Executor::current().now().as_picos();
            refined_port_work(core, true, c.blen * c.mlen, a.clock_period_ps).await;
            {
                let mut s = self.state.lock().unwrap();
                let State { stages, slots, .. } = &mut *s;
                stages[stage]
                    .operands
                    .copy_from_slice(&slots[slot].as_ref().unwrap().tile.as_ref().unwrap().values);
            }
            {
                let mut report = core.report.lock().unwrap();
                let stream = report
                    .refinement
                    .as_mut()
                    .unwrap()
                    .stream_ctrl
                    .as_mut()
                    .unwrap();
                stream.operand_fill_elapsed_ps += Executor::current().now().as_picos() - start;
                let width = c
                    .refinement
                    .as_ref()
                    .unwrap()
                    .weight_read_elements_per_cycle;
                stream.operand_fill_busy_ps += ((c.blen * c.mlen).div_ceil(width) as u64
                    * a.clock_period_ps)
                    .div_ceil(a.diagnostic.weight_port_speedup);
            }
            self.update_peaks(core);
            // Same tile-decoded event, operand-installed phase flag. Ready
            // contexts are exposed only after the actual SRAM read completed.
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

    #[allow(clippy::too_many_arguments)]
    async fn issue_actor(
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
            let has_ready = {
                let s = self.state.lock().unwrap();
                if s.done {
                    return Ok(());
                }
                s.ready.select(s.cursor, mode).is_some()
            };
            if !has_ready {
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
            }
            let (ci, bi, si, slot, m0, mr, nr, kr, n0, k0, lifetime) = {
                let selected_ci = {
                    let s = self.state.lock().unwrap();
                    s.ready.select(s.cursor, mode).unwrap()
                };
                let bi = selected_ci / self.mb;
                let _port = self
                    .access(core, a, 2, true, None, bi, ControlCostClass::BurstStart)
                    .await;
                let mut s = self.state.lock().unwrap();
                let ci = if a.diagnostic.control_ports > 1 {
                    selected_ci
                } else {
                    s.ready
                        .select(s.cursor, mode)
                        .expect("only issuer consumes ready bits")
                };
                let bi = ci / self.mb;
                let mi = ci % self.mb;
                s.ready.set(ci, false);
                s.decoded.set(ci, false);
                s.prev_done.set(ci, false);
                s.cursor = if mode == ReadySelection::BandRotating {
                    ((bi + 1) * self.mb) % s.ready.len
                } else {
                    (ci + 1) % s.ready.len
                };
                let b = s.bands[bi].as_mut().unwrap();
                let slot = b.slot.unwrap();
                let n0 = b.n;
                let k0 = b.k;
                let lifetime = b.lifetime;
                b.pending += 1;
                b.unissued -= 1;
                s.pending += 1;
                assert_eq!(s.records[ci].next_k, k0);
                s.records[ci].next_k += c.mlen;
                let si = s.slots[slot].as_ref().unwrap().stage.unwrap();
                let m0 = mi * r.m_rows;
                (
                    ci,
                    bi,
                    si,
                    slot,
                    m0,
                    r.m_rows.min(m - m0),
                    c.blen.min(n - n0),
                    c.mlen.min(k - k0),
                    n0,
                    k0,
                    lifetime,
                )
            };
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
                                * s.stages[si].operands[col * c.mlen + kk].to_f32();
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
            let completion_pool = self.clone();
            let completion_core = core.clone();
            let clock = a.clock_period_ps;
            ex.spawn(async move {
                Executor::current().resolve_at(result_time).await;
                refined_port_work(&completion_core, false, mr * nr, clock).await;
                completion_pool
                    .send(
                        &completion_core,
                        Event {
                            kind: 2,
                            source: ci as u8,
                            band: bi as u16,
                            mask: 0,
                            lifetime,
                        },
                    )
                    .await;
            });
            {
                let mut report = core.report.lock().unwrap();
                report.useful_macs += (mr * nr * kr) as u64;
                report.issued_macs += (rows * c.blen * c.mlen) as u64;
                report.compute_busy_ps += service;
                let detail = report.refinement.as_mut().unwrap();
                if last_band.is_some_and(|prev| prev != bi) {
                    detail.independent_output_switches += 1;
                }
                detail.output_pool.as_mut().unwrap().context_updates += 1;
                detail.stream_ctrl.as_mut().unwrap().issued_contexts += 1;
            }
            last_band = Some(bi);
            self.update_peaks(core);
            ex.resolve_at(Duration::from_picos(service)).await;
            if self.state.lock().unwrap().bands[bi]
                .as_ref()
                .unwrap()
                .unissued
                == 0
            {
                let _port = self
                    .access(core, a, 3, false, None, bi, ControlCostClass::OperandRetire)
                    .await;
                let mut s = self.state.lock().unwrap();
                s.slots[slot].take(); // Releases packed+decoded reservation, exactly once.
                s.stages[si].slot = None;
                s.free_slots.push_back(slot);
                s.free_stages.push_back(si);
                let b = s.bands[bi].as_mut().unwrap();
                b.slot = None;
                b.k += c.mlen;
                b.unissued = self.mb;
                if b.remaining == 0 {
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
    let config = r.output_pool.as_ref().unwrap();
    let mb = m.div_ceil(r.m_rows);
    let bands = (config.output_contexts / mb).min(region.rows.div_ceil(c.blen));
    if bands == 0 {
        return Err("event pool cannot admit a complete M cohort".into());
    }
    let q = config.output_contexts;
    let pool = Arc::new(Pool {
        state: Mutex::new(State {
            bands: (0..bands).map(|_| None).collect(),
            records: (0..q).map(|_| Record { next_k: 0 }).collect(),
            slots: (0..c.resident_slots()).map(|_| None).collect(),
            stages: (0..config.operand_stages)
                .map(|_| Stage {
                    slot: None,
                    operands: vec![bf16::ZERO; c.blen * c.mlen],
                })
                .collect(),
            free_bands: (0..bands).collect(),
            free_slots: (0..c.resident_slots()).collect(),
            free_stages: (0..config.operand_stages).collect(),
            loads: VecDeque::new(),
            fills: VecDeque::new(),
            ready: Bitmap::new(q),
            decoded: Bitmap::new(q),
            prev_done: Bitmap::new(q),
            next_n: 0,
            live: 0,
            pending: 0,
            cursor: 0,
            lifetime: 0,
            done: false,
            error: None,
        }),
        control: ControlPort::new(a.diagnostic.control_ports, bands),
        events: Mutex::new(VecDeque::new()),
        event_space: Semaphore::new(c.resident_slots()),
        blocked_producers: AtomicUsize::new(0),
        completion_collectors: AtomicUsize::new(0),
        observation: Mutex::new(Observation::default()),
        manager: Notify::new(),
        filler: Notify::new(),
        issuer: Notify::new(),
        mb,
        capacity: c.resident_slots(),
        cohort: c.cohort_control(),
        split: c.split_window().is_some(),
    });
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
    let acc = &mut accumulator[..m * region.rows];
    acc.fill(0.0);
    futures::try_join!(
        pool.events_actor(core, a, shared, region),
        pool.operand_actor(core, a, shared, region),
        async {
            if c.cohort_control() {
                pool.cohort_issue_actor(input, acc, m, region, core, a)
                    .await
            } else {
                pool.issue_actor(input, acc, m, region, core, a).await
            }
        }
    )?;
    assert!(pool.events.lock().unwrap().is_empty());
    assert_eq!(pool.blocked_producers.load(Ordering::SeqCst), 0);
    assert_eq!(pool.completion_collectors.load(Ordering::SeqCst), 0);
    {
        pool.observe_full_waits();
        let observation = pool.observation.lock().unwrap();
        assert!(observation.landing_full_since.is_none() && observation.stage_full_since.is_none());
        core.split_observe(|r| {
            r.landing_slot_wait_ps += observation.landing_full_ps;
            r.operand_stage_wait_ps += observation.stage_full_ps;
        });
        assert!(observation.queued.is_empty());
        let overlap = overlap_ps(&observation.event_service, &observation.mac_service);
        let busy: u64 = observation.event_service.iter().map(|(a, b)| b - a).sum();
        let mut report = core.report.lock().unwrap();
        let stream = report
            .refinement
            .as_mut()
            .unwrap()
            .stream_ctrl
            .as_mut()
            .unwrap();
        stream.event_service_mac_overlap_ps += overlap;
        stream.event_service_no_mac_ps += busy - overlap;
        let (occupancy, peak) = pool.control.occupancy();
        stream.control_occupancy_ps += occupancy;
        stream.control_ports_peak = stream.control_ports_peak.max(peak);
    }
    let start = Executor::current().now().as_picos();
    refined_port_work(core, false, m * region.rows, a.clock_period_ps).await;
    let wait = shared.vector.work(m * region.rows).await;
    {
        let mut report = core.report.lock().unwrap();
        report.vector_wait_ps += wait;
        let detail = report.refinement.as_mut().unwrap();
        detail.finalized_elements += (m * region.rows) as u64;
        detail.output_finalize_elapsed_ps += Executor::current().now().as_picos() - start;
    }
    for (value, sum) in output.iter_mut().zip(acc.iter().copied()) {
        *value = bf16::from_f32(sum);
        if !value.is_finite() {
            return Err("GEMM produced non-finite BF16".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn service_overlap_excludes_queue_wait_and_counts_partial_intersections_once() {
        let events = [(0, 4), (8, 10), (12, 17), (20, 20)];
        let mac = [(2, 9), (10, 14), (16, 22)];
        assert_eq!(overlap_ps(&events, &mac), 2 + 1 + 2 + 1);
        assert_eq!(overlap_ps(&events, &[]), 0);
        assert_eq!(overlap_ps(&mac, &events), 6);
    }

    #[test]
    fn bounded_encoder_wraps_and_event_payload_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Event>(), 16);
        let mut bits = Bitmap::new(256);
        for i in [1, 63, 64, 130, 255] {
            bits.set(i, true);
        }
        assert_eq!(bits.select(65, ReadySelection::Rotating), Some(130));
        assert_eq!(bits.select(65, ReadySelection::Lowest), Some(1));
        bits.set(255, false);
        assert_eq!(bits.select(254, ReadySelection::Rotating), Some(1));
        assert_eq!(bits.select(64, ReadySelection::Rotating), Some(64));
    }
}
