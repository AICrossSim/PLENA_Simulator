//! HBM3 device model for ramulator 2.1.
//!
//! ramulator 2.1's DRAM implementations are data-driven: the C++ `HBM3` class
//! only fixes the level, command and timing-parameter *names* and reads every
//! organisation count, timing value and timing constraint from its config.
//! Upstream generates that config with its Python package
//! (`python/ramulator/dram/hbm3.py` through `DRAMStandard.to_config`); this
//! module reproduces the same serialisation in Rust so the emulator needs no
//! Python at run time. The presets and JEDEC tables below are copied from that
//! file (JESD238), and the unit test pins the output against what the Python
//! generator emits for the same presets.

use serde_json::{Value, json};

/// ramulator models HBM3 with two ticks per CK so that half-cycle row commands
/// can be represented. Every CK-denominated value is converted to ticks on
/// serialisation, exactly as upstream's `to_config` does.
const TICK_MULTIPLIER: u32 = 2;

/// HBM3 primary timing parameters in CK cycles (one JESD238 speed bin).
#[derive(Debug, Clone)]
pub struct HBM3Timing {
    /// Transfer rate in MT/s.
    pub rate: u32,

    // Timing parameters in CK cycles.
    pub n_bl: u32,
    pub n_cl: u32,
    pub n_rcdrd: u32,
    pub n_rcdwr: u32,
    pub n_rp: u32,
    pub n_ras: u32,
    pub n_rc: u32,
    pub n_wr: u32,
    pub n_rtp: u32,
    pub n_cwl: u32,
    pub n_ccds: u32,
    pub n_ccdl: u32,
    pub n_ccdr: u32,
    pub n_rrds: u32,
    pub n_rrdl: u32,
    pub n_wtrs: u32,
    pub n_wtrl: u32,
    pub n_rtw: u32,
    pub n_faw: u32,
    pub n_ppd: u32,
    pub n_rfcpb: u32,
    pub n_rrefd: u32,
    pub n_refi: u32,

    /// Clock period in picoseconds.
    pub tck_ps: u32,
}

impl HBM3Timing {
    /// 6.4 Gb/s per pin: the one HBM3 speed bin ramulator 2.1 ships.
    pub const HBM3_6400MBPS: HBM3Timing = HBM3Timing {
        rate: 6400,
        n_bl: 2,
        n_cl: 20,
        n_rcdrd: 31,
        n_rcdwr: 15,
        n_rp: 26,
        n_ras: 45,
        n_rc: 72,
        n_wr: 33,
        n_rtp: 9,
        n_cwl: 10,
        n_ccds: 2,
        n_ccdl: 4,
        n_ccdr: 3,
        n_rrds: 4,
        n_rrdl: 5,
        n_wtrs: 7,
        n_wtrl: 10,
        n_rtw: 20,
        n_faw: 24,
        n_ppd: 2,
        n_rfcpb: 320,
        n_rrefd: 8,
        n_refi: 6240,
        tck_ps: 625,
    };
}

/// HBM3 organisation (JESD238 Table 4).
#[derive(Debug, Clone)]
pub struct HBM3Org {
    /// Die density in Mb. The per-channel density that selects tRFC is
    /// `density_mb / sid`.
    pub density_mb: u32,
    pub dq: u32,
    pub channel_width: u32,
    pub pseudochannel: u32,
    /// Stack IDs sharing one channel.
    pub sid: u32,
    pub bankgroup: u32,
    pub bank: u32,
    pub row: u32,
    pub column: u32,
}

impl HBM3Org {
    /// 4 Gb die, one stack ID per channel.
    pub const HBM3_4GB: HBM3Org = HBM3Org {
        density_mb: 4096,
        dq: 32,
        channel_width: 32,
        pseudochannel: 2,
        sid: 1,
        bankgroup: 4,
        bank: 4,
        row: 1 << 14,
        column: (1 << 5) << 3,
    };
    /// 8 Gb die, 8-high stack: 4 Gb per channel.
    pub const HBM3_8GB_8HI: HBM3Org = HBM3Org {
        density_mb: 8192,
        sid: 2,
        row: 1 << 13,
        ..Self::HBM3_4GB
    };
    /// 16 Gb die, 8-high stack: 8 Gb per channel.
    pub const HBM3_16GB_8HI: HBM3Org = HBM3Org {
        density_mb: 16384,
        sid: 2,
        row: 1 << 14,
        ..Self::HBM3_4GB
    };
    /// 32 Gb die, 8-high stack: 16 Gb per channel.
    pub const HBM3_32GB_8HI: HBM3Org = HBM3Org {
        density_mb: 32768,
        sid: 2,
        row: 1 << 15,
        ..Self::HBM3_4GB
    };
    /// 32 Gb die, 16-high stack: 8 Gb per channel.
    pub const HBM3_32GB_16HI: HBM3Org = HBM3Org {
        density_mb: 32768,
        sid: 4,
        row: 1 << 15,
        ..Self::HBM3_4GB
    };

    pub const fn channel_density_in_mb(&self) -> u32 {
        self.density_mb / self.sid
    }
}

#[derive(Debug, Clone)]
pub struct HBM3 {
    pub timing: HBM3Timing,
    pub org: HBM3Org,
}

impl serde::Serialize for HBM3 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.resolve_all().serialize(serializer)
    }
}

// Level indices (`HBM3::Level`). Row and column carry no constraints.
const CHANNEL: u32 = 0;
const PSEUDOCHANNEL: u32 = 1;
const SID: u32 = 2;
const BANKGROUP: u32 = 3;
const BANK: u32 = 4;

// Command indices (`HBM3::Command`).
const ACT: u32 = 0;
const PRE_PB: u32 = 1;
const PRE_AB: u32 = 2;
const RD: u32 = 3;
const WR: u32 = 4;
const RDA: u32 = 5;
const WRA: u32 = 6;
const REF_AB: u32 = 7;
const REF_PB: u32 = 8;
const RFM_AB: u32 = 9;
const RFM_PB: u32 = 10;
const COMMAND_COUNT: usize = 11;

/// Ticks each command occupies its command bus (`HBM3::command_cycles`):
/// ACT is 1.5 CK (row-fall-row), the other row commands half a CK, and column
/// commands the default full CK.
const COMMAND_TICKS: [u32; COMMAND_COUNT] = [3, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1];
/// HBM3 has separate row and column command buses.
const ROW_COMMANDS: [u32; 7] = [ACT, PRE_PB, PRE_AB, REF_AB, REF_PB, RFM_AB, RFM_PB];
const COLUMN_COMMANDS: [u32; 4] = [RD, WR, RDA, WRA];

/// One JEDEC timing constraint before command-length adjustment: `latency` CK
/// must separate a `preceding` command from a `following` one at `level`.
struct Constraint {
    level: u32,
    preceding: &'static [u32],
    following: &'static [u32],
    latency: u32,
    /// `latency` is measured from the `window`-th most recent `preceding`
    /// command (the four-activate window of tFAW).
    window: u32,
    /// Also constrain sibling nodes at `level` (tCCDR across stack IDs).
    sibling: bool,
}

const fn cons(
    level: u32,
    preceding: &'static [u32],
    following: &'static [u32],
    latency: u32,
) -> Constraint {
    Constraint {
        level,
        preceding,
        following,
        latency,
        window: 1,
        sibling: false,
    }
}

impl HBM3 {
    /// CK cycles for `ns` nanoseconds at this speed bin, rounded up.
    fn ns_to_ck(&self, ns: u32) -> u32 {
        (ns * 1000).div_ceil(self.timing.tck_ps)
    }

    /// tRFCab from the per-channel density (JESD238 Table 84), in CK.
    // Ordered stair-step over the density table: `match` takes the first arm.
    #[allow(clippy::match_overlapping_arm)]
    fn resolve_n_rfc(&self) -> u32 {
        let t_rfc_ns = match self.org.channel_density_in_mb() {
            ..=4096 => 260,
            ..=8192 => 350,
            ..=16384 => 450,
            _ => 550,
        };
        self.ns_to_ck(t_rfc_ns)
    }

    /// tREFIpb = tREFI (3.9 us) / banks per channel, in CK.
    fn resolve_n_refipb(&self) -> u32 {
        const T_REFI_PS: u32 = 3_900_000;
        let banks_per_channel = self.org.bank * self.org.bankgroup * self.org.sid;
        T_REFI_PS.div_ceil(banks_per_channel * self.timing.tck_ps)
    }

    fn resolve_all(&self) -> Value {
        let t = &self.timing;
        // Secondary timings are resolved in CK; everything is scaled to ticks
        // afterwards, as upstream does.
        let n_rfc = self.resolve_n_rfc();
        let n_rfmab = n_rfc;
        let n_rfmpb = t.n_rfcpb;
        let n_refipb = self.resolve_n_refipb();

        // `HBM3::Timing` order. CK values become ticks and the period halves.
        let ck_timings = [
            t.n_bl, t.n_cl, t.n_rcdrd, t.n_rcdwr, t.n_rp, t.n_ras, t.n_rc, t.n_wr, t.n_rtp,
            t.n_cwl, t.n_ccds, t.n_ccdl, t.n_ccdr, t.n_rrds, t.n_rrdl, t.n_wtrs, t.n_wtrl, t.n_rtw,
            t.n_faw, t.n_ppd, n_rfc, t.n_rfcpb, n_rfmab, n_rfmpb, t.n_rrefd, t.n_refi, n_refipb,
        ];
        let mut timing_params = vec![t.rate];
        timing_params.extend(ck_timings.iter().map(|&ck| ck * TICK_MULTIPLIER));
        timing_params.push(t.tck_ps / TICK_MULTIPLIER);

        let table = [
            // PseudoChannel: independent per pseudo channel.
            cons(PSEUDOCHANNEL, &[RD, RDA], &[RD, RDA], t.n_bl),
            cons(PSEUDOCHANNEL, &[WR, WRA], &[WR, WRA], t.n_bl),
            cons(PSEUDOCHANNEL, &[RD, RDA], &[WR, WRA], t.n_rtw),
            cons(
                PSEUDOCHANNEL,
                &[WR, WRA],
                &[RD, RDA],
                t.n_cwl + t.n_bl + t.n_wtrs,
            ),
            cons(PSEUDOCHANNEL, &[RD], &[PRE_AB], t.n_rtp),
            cons(PSEUDOCHANNEL, &[WR], &[PRE_AB], t.n_cwl + t.n_bl + t.n_wr),
            cons(PSEUDOCHANNEL, &[ACT], &[ACT], t.n_rrds),
            Constraint {
                window: 4,
                ..cons(PSEUDOCHANNEL, &[ACT], &[ACT], t.n_faw)
            },
            cons(PSEUDOCHANNEL, &[ACT], &[PRE_AB], t.n_ras),
            cons(PSEUDOCHANNEL, &[PRE_AB], &[ACT], t.n_rp),
            cons(PSEUDOCHANNEL, &[PRE_PB, PRE_AB], &[PRE_PB, PRE_AB], t.n_ppd),
            cons(PSEUDOCHANNEL, &[ACT], &[REF_AB], t.n_rc),
            cons(PSEUDOCHANNEL, &[PRE_PB, PRE_AB], &[REF_AB], t.n_rp),
            cons(PSEUDOCHANNEL, &[RDA], &[REF_AB], t.n_rp + t.n_rtp),
            cons(
                PSEUDOCHANNEL,
                &[WRA],
                &[REF_AB],
                t.n_cwl + t.n_bl + t.n_wr + t.n_rp,
            ),
            cons(PSEUDOCHANNEL, &[REF_AB], &[ACT, PRE_AB], n_rfc),
            cons(PSEUDOCHANNEL, &[REF_PB], &[REF_PB], t.n_rrefd),
            cons(PSEUDOCHANNEL, &[REF_PB], &[ACT], t.n_rrefd),
            cons(PSEUDOCHANNEL, &[ACT], &[REF_PB], t.n_rrds),
            cons(PSEUDOCHANNEL, &[ACT], &[RFM_AB], t.n_rc),
            cons(PSEUDOCHANNEL, &[PRE_PB, PRE_AB], &[RFM_AB], t.n_rp),
            cons(PSEUDOCHANNEL, &[RDA], &[RFM_AB], t.n_rp + t.n_rtp),
            cons(
                PSEUDOCHANNEL,
                &[WRA],
                &[RFM_AB],
                t.n_cwl + t.n_bl + t.n_wr + t.n_rp,
            ),
            cons(PSEUDOCHANNEL, &[RFM_AB], &[ACT, PRE_AB], n_rfmab),
            cons(PSEUDOCHANNEL, &[RFM_PB], &[ACT], t.n_rrefd),
            cons(PSEUDOCHANNEL, &[ACT], &[RFM_PB], t.n_rrds),
            // Sid: same and sibling stack IDs.
            cons(SID, &[RD, RDA], &[RD, RDA], t.n_ccds),
            cons(SID, &[WR, WRA], &[WR, WRA], t.n_ccds),
            Constraint {
                sibling: true,
                ..cons(SID, &[RD, RDA], &[RD, RDA], t.n_ccdr)
            },
            // BankGroup.
            cons(BANKGROUP, &[RD, RDA], &[RD, RDA], t.n_ccdl),
            cons(BANKGROUP, &[WR, WRA], &[WR, WRA], t.n_ccdl),
            cons(
                BANKGROUP,
                &[WR, WRA],
                &[RD, RDA],
                t.n_cwl + t.n_bl + t.n_wtrl,
            ),
            cons(BANKGROUP, &[ACT], &[ACT], t.n_rrdl),
            // Bank.
            cons(BANK, &[ACT], &[ACT], t.n_rc),
            cons(BANK, &[ACT], &[RD, RDA], t.n_rcdrd),
            cons(BANK, &[ACT], &[WR, WRA], t.n_rcdwr),
            cons(BANK, &[ACT], &[PRE_PB], t.n_ras),
            cons(BANK, &[PRE_PB], &[ACT], t.n_rp),
            cons(BANK, &[RD], &[PRE_PB], t.n_rtp),
            cons(BANK, &[WR], &[PRE_PB], t.n_cwl + t.n_bl + t.n_wr),
            cons(BANK, &[RDA], &[ACT], t.n_rtp + t.n_rp),
            cons(BANK, &[WRA], &[ACT], t.n_cwl + t.n_bl + t.n_wr + t.n_rp),
            cons(BANK, &[REF_PB], &[ACT], t.n_rfcpb),
            cons(BANK, &[ACT], &[REF_PB], t.n_rc),
            cons(BANK, &[PRE_PB], &[REF_PB], t.n_rp),
            cons(BANK, &[RFM_PB], &[ACT], n_rfmpb),
            cons(BANK, &[ACT], &[RFM_PB], t.n_rc),
            cons(BANK, &[PRE_PB], &[RFM_PB], t.n_rp),
        ];
        let mut timing_constraints = bus_occupancy_constraints();
        timing_constraints.extend(expand_constraints(&table));

        json!({
            "channel_width": self.org.channel_width,
            "org": {
                "dq": self.org.dq,
                "count": [
                    1,
                    self.org.pseudochannel,
                    self.org.sid,
                    self.org.bankgroup,
                    self.org.bank,
                    self.org.row,
                    self.org.column,
                ],
            },
            "timing": timing_params,
            "command_cycles": COMMAND_TICKS,
            "read_latency": (t.n_cl + t.n_bl) * TICK_MULTIPLIER,
            "timing_constraints": timing_constraints,
        })
    }
}

/// Expand the JEDEC table into ramulator's
/// `[level, preceding, following, latency(, window(, sibling))]` entries.
///
/// ramulator records a command at its first bus tick while JEDEC measures
/// from its last, so a multi-tick preceding command adds `ticks - 1` and a
/// multi-tick following command subtracts it. Pairs whose adjusted latency
/// differs are split into separate entries, in first-seen order with the
/// following commands sorted, matching `DRAMStandard.to_config`.
fn expand_constraints(table: &[Constraint]) -> Vec<Value> {
    let mut entries = Vec::new();
    for c in table {
        let nominal = i64::from(c.latency * TICK_MULTIPLIER);
        // (adjusted latency, preceding ids, following ids), in first-seen order.
        let mut groups: Vec<(i64, Vec<u32>, Vec<u32>)> = Vec::new();
        for &p in c.preceding {
            let p_off = i64::from(COMMAND_TICKS[p as usize]) - 1;
            for &f in c.following {
                let f_off = i64::from(COMMAND_TICKS[f as usize]) - 1;
                let adjusted = nominal + p_off - f_off;
                let group = match groups.iter().position(|g| g.0 == adjusted) {
                    Some(i) => &mut groups[i],
                    None => {
                        groups.push((adjusted, Vec::new(), Vec::new()));
                        groups.last_mut().unwrap()
                    }
                };
                if !group.1.contains(&p) {
                    group.1.push(p);
                }
                if !group.2.contains(&f) {
                    group.2.push(f);
                }
            }
        }
        for (latency, preceding, mut following) in groups {
            following.sort_unstable();
            let mut entry = vec![
                json!(c.level),
                json!(preceding),
                json!(following),
                json!(latency),
            ];
            if c.window != 1 || c.sibling {
                entry.push(json!(c.window));
            }
            if c.sibling {
                entry.push(json!(true));
            }
            entries.push(Value::Array(entry));
        }
    }
    entries
}

/// Channel-level bus-occupancy constraints: a command that holds its (row or
/// column) bus for more than one tick blocks every command on that bus for
/// that long.
fn bus_occupancy_constraints() -> Vec<Value> {
    let mut entries = Vec::new();
    for bus in [&ROW_COMMANDS[..], &COLUMN_COMMANDS[..]] {
        let mut all = bus.to_vec();
        all.sort_unstable();
        // (ticks, commands), in first-seen order.
        let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
        for &cmd in bus {
            let ticks = COMMAND_TICKS[cmd as usize];
            match groups.iter_mut().find(|g| g.0 == ticks) {
                Some(group) => group.1.push(cmd),
                None => groups.push((ticks, vec![cmd])),
            }
        }
        for (ticks, commands) in groups {
            if ticks == 1 {
                continue;
            }
            entries.push(json!([CHANNEL, commands, all, ticks]));
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DRAM;

    /// The expected values are what upstream's generator emits for the same
    /// presets: `HBM3(org_preset="HBM3_8Gb_8hi", timing_preset="HBM3_6400Mbps")
    /// .to_config()` in ramulator 2.1's Python package.
    #[test]
    fn hbm3_8gb_8hi_6400_matches_upstream_generator() {
        let hbm3 = HBM3 {
            timing: HBM3Timing::HBM3_6400MBPS,
            org: HBM3Org::HBM3_8GB_8HI,
        };
        let v = serde_json::to_value(DRAM::HBM3(hbm3)).unwrap();

        assert_eq!(v["impl"], "HBM3");
        assert_eq!(v["channel_width"], 32);
        assert_eq!(
            v["org"],
            json!({"dq": 32, "count": [1, 2, 2, 4, 4, 8192, 256]})
        );
        assert_eq!(
            v["timing"],
            json!([
                6400, 4, 40, 62, 30, 52, 90, 144, 66, 18, 20, 4, 8, 6, 8, 10, 14, 20, 40, 48, 4,
                832, 640, 832, 640, 16, 12480, 390, 312
            ])
        );
        assert_eq!(
            v["command_cycles"],
            json!([3, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1])
        );
        assert_eq!(v["read_latency"], 44);

        let cons = v["timing_constraints"].as_array().unwrap();
        assert_eq!(cons.len(), 52);
        // Bus occupancy comes first: ACT holds the row bus for three ticks,
        // column commands hold theirs for two.
        assert_eq!(cons[0], json!([0, [0], [0, 1, 2, 7, 8, 9, 10], 3]));
        assert_eq!(cons[1], json!([0, [3, 4, 5, 6], [3, 4, 5, 6], 2]));
        // Command-length adjustment: the one-tick column offsets cancel in the
        // WR -> RD turnaround (nCWL + nBL + nWTRS = 38), PREab -> ACT loses
        // ACT's two extra ticks, and REFab -> {ACT, PREab} splits in two.
        assert!(cons.contains(&json!([1, [4, 6], [3, 5], 38])));
        assert!(cons.contains(&json!([1, [2], [0], 50])));
        assert!(cons.contains(&json!([1, [7], [0], 830])));
        assert!(cons.contains(&json!([1, [7], [2], 832])));
        // The tFAW window and the sibling-SID tCCDR flag are carried through.
        assert!(cons.contains(&json!([1, [0], [0], 48, 4])));
        assert!(cons.contains(&json!([2, [3, 5], [3, 5], 6, 1, true])));
        assert_eq!(cons[cons.len() - 2], json!([4, [0], [10], 146]));
        assert_eq!(cons[cons.len() - 1], json!([4, [1], [10], 52]));
    }
}
