//! Packed-only landing storage; actual bytes decode directly into a paid stage.
use super::*;

impl CoreState {
    pub(super) fn split_observe(&self, update: impl FnOnce(&mut SplitWindowReport)) {
        if self.config.split_window().is_some() {
            let mut r = self.report.lock().unwrap();
            update(
                r.refinement
                    .as_mut()
                    .unwrap()
                    .split_window
                    .as_mut()
                    .unwrap(),
            );
        }
    }
    pub(super) fn change_lifetime(&self, packed: i64, operand: i64, decoding: i64) {
        if self.config.split_window().is_none() {
            return;
        }
        let at = Executor::current().now().as_picos();
        self.split_observe(|r| {
            for (current, delta) in r
                .live_current_bytes
                .iter_mut()
                .zip([packed, operand, decoding])
            {
                *current = if delta >= 0 {
                    current.checked_add(delta as u64).unwrap()
                } else {
                    current.checked_sub(delta.unsigned_abs()).unwrap()
                };
            }
            let [p, o, d] = r.live_current_bytes;
            assert!(
                p <= (r.packed_slots * r.packed_slot_bytes) as u64
                    && o + d <= (2 * r.operand_stage_bytes) as u64
            );
            assert!(p + o + d <= self.config.weight_sram_bytes as u64);
            if p + o + d > r.live_peak_bytes {
                r.live_peak_bytes = p + o + d;
                r.live_peak_classes = [p, o, d];
            }
            r.lifetime_changes.push([at, p, o, d]);
        });
    }
}

pub(super) struct PackedTile {
    pub bytes: Vec<u8>,
    pub issued_ps: u64,
    pub arrived_ps: u64,
    pub nr: usize,
    pub kr: usize,
    pub _reservation: SlotReservation,
}
impl PackedTile {
    pub fn decode_into(&self, destination: &mut [bf16], c: &CoreConfig) -> Result<(), String> {
        assert_eq!(destination.len(), c.blen * c.mlen);
        destination.fill(bf16::ZERO);
        let elements = c.blen * c.mlen;
        let element_type = DataType::Fp(FpType {
            sign: true,
            exponent: 4,
            mantissa: 3,
        });
        let scale_type = DataType::Fp(FpType::E8M0);
        for row in 0..self.nr {
            for k in 0..self.kr {
                let element = element_type.convert_bits_to_f32(self.bytes[row * c.mlen + k] as u32);
                let scale = scale_type.convert_bits_to_f32(
                    self.bytes[elements + row * (c.mlen / BLOCK) + k / BLOCK] as u32,
                );
                let value = bf16::from_f32(element * scale);
                if !value.is_finite() {
                    return Err("decoded weight is not finite BF16".into());
                }
                destination[row * c.mlen + k] = value;
            }
        }
        Ok(())
    }
}

pub(super) fn spawn_packed_load(
    core: Arc<CoreState>,
    shared: Arc<Shared>,
    region: MatrixRegion,
    tile: TileSpec,
) -> oneshot::Receiver<Result<PackedTile, String>> {
    let reservation = SlotReservation::new(core.clone());
    let (tx, rx) = oneshot::channel();
    Executor::current().spawn(async move {
        let _ = tx.send(load_packed(core, shared, region, tile, reservation).await);
    });
    rx
}
async fn load_packed(
    core: Arc<CoreState>,
    shared: Arc<Shared>,
    region: MatrixRegion,
    tile: TileSpec,
    reservation: SlotReservation,
) -> Result<PackedTile, String> {
    let issued_ps = Executor::current().now().as_picos();
    let c = &core.config;
    let nr = c.blen.min(region.rows - tile.n);
    let kr = c.mlen.min(region.cols - tile.k);
    let elements = c.blen * c.mlen;
    let packed = Arc::new(Mutex::new(vec![0u8; elements + elements / BLOCK]));
    let mut reads = BTreeMap::new();
    for row in 0..nr {
        add_reads(
            &mut reads,
            region.element_base + (tile.n + row) as u64 * region.element_row_stride + tile.k as u64,
            kr,
            row * c.mlen,
        );
        add_reads(
            &mut reads,
            region.scale_base
                + (tile.n + row) as u64 * region.scale_row_stride
                + (tile.k / BLOCK) as u64,
            kr.div_ceil(BLOCK),
            elements + row * (c.mlen / BLOCK),
        );
    }
    let mut pending = Vec::with_capacity(reads.len());
    for (address, spans) in reads {
        let mut mask = 0u8;
        for span in &spans {
            mask |= 1 << (span.src / 32);
            mask |= 1 << ((span.src + span.len - 1) / 32);
        }
        let dma = shared.dma.clone();
        let core = core.clone();
        let packed = packed.clone();
        let reserved = dma.reserved_pool.as_ref().map(|p| {
            p.reserve(
                dma.core_indices[&core.config.id],
                u64::from(mask.count_ones()) * 32,
            )
        });
        let (tx, rx) = oneshot::channel();
        pending.push(rx);
        Executor::current().spawn(async move {
            let start = Executor::current().now().as_picos();
            let (credit, fair, weighted) = if let Some(p) = &dma.reserved_pool {
                (
                    None,
                    None,
                    Some(p.acquire(dma.core_indices[&core.config.id]).await),
                )
            } else if let Some(p) = &dma.fair_pool {
                (
                    None,
                    Some(p.acquire(dma.core_indices[&core.config.id]).await),
                    None,
                )
            } else {
                (Some(dma.credits.acquire().await.unwrap()), None, None)
            };
            core.split_observe(|r| {
                r.dma_credit_wait_ps += Executor::current().now().as_picos() - start
            });
            let current = dma.inflight.fetch_add(1, Ordering::SeqCst) + 1;
            dma.peak.fetch_max(current, Ordering::SeqCst);
            let bytes = if dma.config.is_some() {
                dma.read(
                    address,
                    mask,
                    spans.iter().map(|s| s.len).sum(),
                    core.clone(),
                    None,
                )
                .await
            } else {
                let bytes = dma.hbm.box_read(address).await;
                dma.bytes.fetch_add(64, Ordering::SeqCst);
                core.report.lock().unwrap().hbm_read_bytes += 64;
                bytes
            };
            {
                let mut buffer = packed.lock().unwrap();
                for s in spans {
                    buffer[s.dst..s.dst + s.len].copy_from_slice(&bytes[s.src..s.src + s.len]);
                }
            }
            dma.inflight.fetch_sub(1, Ordering::SeqCst);
            drop(reserved);
            drop(credit);
            drop(fair);
            drop(weighted);
            let _ = tx.send(());
        });
    }
    for rx in pending {
        rx.await.map_err(|_| "packed DMA request failed")?;
    }
    let arrived_ps = Executor::current().now().as_picos();
    core.projection_observed
        .lock()
        .unwrap()
        .first_tile_arrival_ps
        .get_or_insert(arrived_ps);
    core.split_observe(|r| record_load(&mut r.packed_arrivals, arrived_ps - issued_ps));
    let bytes = Arc::try_unwrap(packed)
        .map_err(|_| "packed bytes retained after DMA")?
        .into_inner()
        .unwrap();
    Ok(PackedTile {
        bytes,
        issued_ps,
        arrived_ps,
        nr,
        kr,
        _reservation: reservation,
    })
}
