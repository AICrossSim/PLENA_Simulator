//! Explicit coefficient registers and one bounded 32-row sector, no cache.
use super::Accelerator;
use crate::v2_timing::Calendar;
use quantize::tensor_to_f32_vec;
use sram::matrix::MatrixLayout;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub(super) struct CoefficientView {
    base: u32,
    head_stride: u32,
    row_stride: u32,
    repeat: u32,
    pub heads: u32,
    extent: u32,
}
impl CoefficientView {
    pub(super) fn unpack(w: u64) -> Self {
        assert_eq!(w >> 62, 0, "reserved coefficient bits");
        let v = Self {
            base: (w & 0x7ffff) as u32,
            head_stride: ((w >> 19) & 8191) as u32,
            row_stride: ((w >> 32) & 511) as u32,
            repeat: ((w >> 41) & 7) as u32,
            heads: ((w >> 44) & 63) as u32 + 1,
            extent: ((w >> 50) & 4095) as u32 + 1,
        };
        assert!(v.base + v.extent <= 524288, "coefficient SRAM capacity");
        v
    }
    fn address(self, row: u32, head: u32) -> u32 {
        assert!(head < self.heads);
        let offset = (head >> self.repeat) * self.head_stride + row * self.row_stride;
        assert!(offset < self.extent, "coefficient extent violation");
        self.base + offset
    }
}

pub(super) struct Sector {
    views: Vec<CoefficientView>,
    words: BTreeMap<u32, Vec<f32>>,
    pub ready: u64,
}
impl Sector {
    pub fn value(&self, field: usize, row: u32, head: u32) -> f32 {
        let address = self.views[field].address(row, head);
        self.words[&(address / 32)][(address % 32) as usize]
    }
}

impl Accelerator {
    pub(super) async fn native_sector(
        &mut self,
        fields: &[usize],
        row: u32,
        rows: u32,
        heads: u32,
        ready: u64,
        clock: &mut Calendar,
    ) -> Sector {
        assert!(fields.len() <= 2 && row.is_multiple_of(32));
        let views: Vec<_> = fields
            .iter()
            .map(|&i| self.native_coeff[i].expect("native EXEC without CCFG"))
            .collect();
        let mut words = BTreeSet::new();
        for view in &views {
            assert_eq!(heads, view.heads, "native and state head count differ");
            // The first implementation has regular sector refill counters,
            // not a dynamic arbitrary gather/deduplication network.
            assert!(
                (view.row_stride == 0 && view.head_stride <= 1)
                    || (view.row_stride == 1
                        && (view.head_stride == 0 || view.head_stride == 128)
                        && view.base.is_multiple_of(32)),
                "unsupported native sector geometry"
            );
            for h in 0..heads {
                for r in row..(row + 32).min(rows) {
                    words.insert(view.address(r, h) / 32);
                }
            }
        }
        assert!(words.len() <= 64, "sector exceeds bounded 4 KiB staging");
        // One word address per cycle from the incremental AGU. Refill and
        // selector are charged, even when coefficient bits are already local.
        let address_ready = ready + words.len() as u64;
        let layout = MatrixLayout {
            rows: 1,
            cols: 32,
            tile_count: 1,
            tile_pitch_rows: 1,
            alpha: 1,
            tile_skew: 0,
        };
        let requests: Vec<_> = words.iter().map(|&w| (w * 32, layout)).collect();
        let (packets, service) = self.m_machine.mram.read_layout_packets(&requests).await;
        let port = std::env::var("PLENA_V2_SRAM_PORT").map_or(1, |s| s.parse::<u64>().unwrap());
        let end = clock.matrix(
            address_ready,
            service.service_cycles * port,
            service.bank_words,
            false,
        ) + 1; // registered selection/broadcast
        clock.end = clock.end.max(end);
        Sector {
            views,
            words: words
                .into_iter()
                .zip(
                    packets
                        .into_iter()
                        .map(|q| tensor_to_f32_vec(q.as_tensor())),
                )
                .collect(),
            ready: end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offset_shared_and_strided_addresses() {
        let w = 300032 | (128u64 << 19) | (1 << 32) | (3 << 41) | (31 << 44) | (511 << 50);
        let v = CoefficientView::unpack(w);
        assert_eq!(v.address(127, 31), 300543);
        assert_eq!(v.address(17, 0), v.address(17, 7));
    }
    #[test]
    #[should_panic(expected = "extent")]
    fn out_of_extent_is_not_silently_wrapped() {
        CoefficientView::unpack(1 << 32).address(1, 0);
    }
    #[test]
    #[should_panic(expected = "reserved")]
    fn reserved_bits_rejected() {
        CoefficientView::unpack(1 << 63);
    }
}
