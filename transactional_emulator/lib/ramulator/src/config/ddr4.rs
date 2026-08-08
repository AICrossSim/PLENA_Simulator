/// DDR4 timing parameters.
#[derive(Debug, Clone)]
pub struct DDR4Timing {
    /// Transfer rate in MT/s.
    pub rate: u32,

    // Timing parameters in cycles.
    pub n_bl: u32,
    pub n_cl: u32,
    pub n_rcd: u32,
    pub n_rp: u32,
    pub n_ras: u32,
    pub n_rc: u32,
    pub n_wr: u32,
    pub n_rtp: u32,
    pub n_cwl: u32,
    pub n_ccds: u32,
    pub n_ccdl: u32,
    pub n_wtrs: u32,
    pub n_wtrl: u32,
    pub n_cs: u32,
}

impl DDR4Timing {
    pub const DDR4_2400P: DDR4Timing = DDR4Timing {
        rate: 2400,
        n_bl: 4,
        n_cl: 15,
        n_rcd: 15,
        n_rp: 15,
        n_ras: 39,
        n_rc: 54,
        n_wr: 18,
        n_rtp: 9,
        n_cwl: 12,
        n_ccds: 4,
        n_ccdl: 6,
        n_wtrs: 3,
        n_wtrl: 9,
        n_cs: 2,
    };
    pub const DDR4_2400R: DDR4Timing = DDR4Timing {
        n_cl: 16,
        n_rcd: 16,
        n_rp: 16,
        n_rc: 55,
        ..Self::DDR4_2400P
    };
    pub const DDR4_2400U: DDR4Timing = DDR4Timing {
        n_cl: 17,
        n_rcd: 17,
        n_rp: 17,
        n_rc: 56,
        ..Self::DDR4_2400P
    };
    pub const DDR4_2400T: DDR4Timing = DDR4Timing {
        n_cl: 18,
        n_rcd: 18,
        n_rp: 18,
        n_rc: 57,
        ..Self::DDR4_2400P
    };
}

#[derive(Debug, Clone)]
pub struct DDR4Org {
    pub dq: u32,
    pub rank: u32,
    pub bankgroup: u32,
    pub bank: u32,
    pub row: u32,
    pub column: u32,
}

impl DDR4Org {
    pub const DDR4_2GB_X4: DDR4Org = DDR4Org {
        dq: 4,
        rank: 1,
        bankgroup: 4,
        bank: 4,
        row: 1 << 15,
        column: 1 << 10,
    };
    pub const DDR4_2GB_X8: DDR4Org = DDR4Org {
        dq: 8,
        row: 1 << 14,
        ..Self::DDR4_2GB_X4
    };
    pub const DDR4_2GB_X16: DDR4Org = DDR4Org {
        dq: 16,
        bankgroup: 2,
        row: 1 << 14,
        ..Self::DDR4_2GB_X4
    };

    pub const DDR4_4GB_X4: DDR4Org = DDR4Org {
        row: 1 << 16,
        ..Self::DDR4_2GB_X4
    };
    pub const DDR4_4GB_X8: DDR4Org = DDR4Org {
        dq: 8,
        row: 1 << 15,
        ..Self::DDR4_4GB_X4
    };
    pub const DDR4_4GB_X16: DDR4Org = DDR4Org {
        dq: 16,
        bankgroup: 2,
        row: 1 << 15,
        ..Self::DDR4_4GB_X4
    };

    pub const DDR4_8GB_X4: DDR4Org = DDR4Org {
        row: 1 << 17,
        ..Self::DDR4_2GB_X4
    };
    pub const DDR4_8GB_X8: DDR4Org = DDR4Org {
        dq: 8,
        row: 1 << 16,
        ..Self::DDR4_8GB_X4
    };
    pub const DDR4_8GB_X16: DDR4Org = DDR4Org {
        dq: 16,
        bankgroup: 2,
        row: 1 << 16,
        ..Self::DDR4_8GB_X4
    };

    pub const DDR4_16GB_X4: DDR4Org = DDR4Org {
        row: 1 << 18,
        ..Self::DDR4_2GB_X4
    };
    pub const DDR4_16GB_X8: DDR4Org = DDR4Org {
        dq: 8,
        row: 1 << 17,
        ..Self::DDR4_16GB_X4
    };
    pub const DDR4_16GB_X16: DDR4Org = DDR4Org {
        dq: 16,
        bankgroup: 2,
        row: 1 << 17,
        ..Self::DDR4_16GB_X4
    };

    pub const fn density_in_mb(&self) -> u32 {
        (((self.dq * self.rank * self.bankgroup * self.bank) as u64
            * self.row as u64
            * self.column as u64)
            / 1024
            / 1024) as u32
    }
}

const _: () = {
    assert!(DDR4Org::DDR4_4GB_X4.density_in_mb() == 4096);
};

#[derive(Debug, Clone)]
pub struct DDR4 {
    pub timing: DDR4Timing,
    pub org: DDR4Org,
}

impl serde::Serialize for DDR4 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.resolve_all().serialize(serializer)
    }
}

impl DDR4 {
    fn ns_to_cycle(&self, ns: u32) -> u32 {
        (ns * self.timing.rate).div_ceil(2000)
    }

    // Ordered stair-step over JEDEC speed bins: `match` takes the first arm.
    #[allow(clippy::match_overlapping_arm)]
    fn resolve_n_rrds(&self) -> u32 {
        match self.org.dq {
            4 | 8 => 4,
            16 => match self.timing.rate {
                ..=1866 => 5,
                ..=2133 => 6,
                ..=2400 => 7,
                ..=2933 => 8,
                ..=3200 => 9,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    // Ordered stair-step over JEDEC speed bins: `match` takes the first arm.
    #[allow(clippy::match_overlapping_arm)]
    fn resolve_n_rrdl(&self) -> u32 {
        match self.org.dq {
            4 | 8 => match self.timing.rate {
                ..=1866 => 5,
                ..=2400 => 6,
                ..=2666 => 7,
                ..=3200 => 8,
                _ => unreachable!(),
            },
            16 => match self.timing.rate {
                ..=1866 => 6,
                ..=2133 => 7,
                ..=2400 => 8,
                ..=2666 => 9,
                ..=2933 => 10,
                ..=3200 => 11,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    // Ordered stair-step over JEDEC speed bins: `match` takes the first arm.
    #[allow(clippy::match_overlapping_arm)]
    fn resolve_n_faw(&self) -> u32 {
        match self.org.dq {
            4 => 16,
            8 => match self.timing.rate {
                ..=1600 => 20,
                ..=1866 => 22,
                ..=2133 => 23,
                ..=2400 => 26,
                ..=2666 => 28,
                ..=2933 => 31,
                ..=3200 => 34,
                _ => unreachable!(),
            },
            16 => match self.timing.rate {
                ..=1866 => 28,
                ..=2133 => 32,
                ..=2400 => 36,
                ..=2666 => 40,
                ..=2933 => 44,
                ..=3200 => 48,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    // Ordered stair-step over JEDEC speed bins: `match` takes the first arm.
    #[allow(clippy::match_overlapping_arm)]
    fn resolve_n_rfc(&self) -> u32 {
        let t_rfc = match self.org.density_in_mb() {
            ..=2048 => 160,
            ..=4096 => 260,
            ..=8192 => 360,
            ..=16384 => 550,
            _ => unreachable!(),
        };

        self.ns_to_cycle(t_rfc)
    }

    fn resolve_all(&self) -> serde_json::Value {
        let DDR4Timing {
            rate,
            n_bl,
            n_cl,
            n_rcd,
            n_rp,
            n_ras,
            n_rc,
            n_wr,
            n_rtp,
            n_cwl,
            n_ccds,
            n_ccdl,
            n_wtrs,
            n_wtrl,
            n_cs,
        } = self.timing;

        let n_rrds = self.resolve_n_rrds();
        let n_rrdl = self.resolve_n_rrdl();
        let n_faw = self.resolve_n_faw();
        let n_rfc = self.resolve_n_rfc();
        let n_refi = self.ns_to_cycle(7800);

        let tck_ps = 2e6 / rate as f32;

        let timing_params = serde_json::json!([
            rate, n_bl, n_cl, n_rcd, n_rp, n_ras, n_rc, n_wr, n_rtp, n_cwl, n_ccds, n_ccdl, n_rrds,
            n_rrdl, n_wtrs, n_wtrl, n_faw, n_rfc, n_refi, n_cs, tck_ps
        ]);

        const CHANNEL: u32 = 0;
        const RANK: u32 = 1;
        const BANKGROUP: u32 = 2;
        const BANK: u32 = 3;

        const ACT: u32 = 0;
        const PRE_PB: u32 = 1;
        const PRE_AB: u32 = 2;
        const RD: u32 = 3;
        const WR: u32 = 4;
        const RDA: u32 = 5;
        const WRA: u32 = 6;
        const REF_AB: u32 = 7;

        let timing_constraints = serde_json::json!([
            [CHANNEL, [RD, RDA], [RD, RDA], n_bl],
            [CHANNEL, [WR, WRA], [WR, WRA], n_bl],
            [RANK, [RD, RDA], [RD, RDA], n_ccds],
            [RANK, [WR, WRA], [WR, WRA], n_ccds],
            [
                RANK,
                [RD, RDA],
                [WR, WRA],
                (n_cl + n_bl + 2).saturating_sub(n_cwl)
            ],
            [RANK, [WR, WRA], [RD, RDA], n_cwl + n_bl + n_wtrs],
            [RANK, [RD, RDA], [RD, RDA, WR, WRA], n_bl + n_cs, 1, true],
            [
                RANK,
                [WR, WRA],
                [RD, RDA],
                (n_cwl + n_bl + n_cs).saturating_sub(n_cl),
                1,
                true
            ],
            [RANK, [RD], [PRE_AB], n_rtp],
            [RANK, [WR], [PRE_AB], n_cwl + n_bl + n_wr],
            [RANK, [ACT], [ACT], n_rrds],
            [RANK, [ACT], [ACT], n_faw, 4],
            [RANK, [ACT], [PRE_AB], n_ras],
            [RANK, [PRE_AB], [ACT], n_rp],
            [RANK, [ACT], [REF_AB], n_rc],
            [RANK, [PRE_PB, PRE_AB], [REF_AB], n_rp],
            [RANK, [RDA], [REF_AB], n_rp + n_rtp],
            [RANK, [WRA], [REF_AB], n_cwl + n_bl + n_wr + n_rp],
            [RANK, [REF_AB], [ACT, PRE_AB], n_rfc],
            [BANKGROUP, [RD, RDA], [RD, RDA], n_ccdl],
            [BANKGROUP, [WR, WRA], [WR, WRA], n_ccdl],
            [BANKGROUP, [WR, WRA], [RD, RDA], n_cwl + n_bl + n_wtrl],
            [BANKGROUP, [ACT], [ACT], n_rrdl],
            [BANK, [ACT], [ACT], n_rc],
            [BANK, [ACT], [RD, RDA, WR, WRA], n_rcd],
            [BANK, [ACT], [PRE_PB], n_ras],
            [BANK, [PRE_PB], [ACT], n_rp],
            [BANK, [RD], [PRE_PB], n_rtp],
            [BANK, [WR], [PRE_PB], n_cwl + n_bl + n_wr],
            [BANK, [RDA], [ACT], n_rtp + n_rp],
            [BANK, [WRA], [ACT], n_cwl + n_bl + n_wr + n_rp],
        ]);

        serde_json::json!({
            "channel_width": 64,
            "org": {
                "dq": self.org.dq,
                "count": [1, self.org.rank, self.org.bankgroup, self.org.bank, self.org.row, self.org.column],
            },
            "timing": timing_params,
            "read_latency": n_cl + n_bl,
            "timing_constraints": timing_constraints,
        })
    }
}
