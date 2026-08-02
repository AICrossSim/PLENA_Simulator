use std::cmp::max;

/// HBM2 timing parameters.
#[derive(Debug, Clone)]
pub struct HBM2Timing {
    /// Transfer rate in MT/s.
    pub rate: u32,

    // Timing parameters in cycles.
    pub n_bl: u32,
    pub n_cl: u32,
    pub n_rcdrd: u32,
    pub n_rcdwr: u32,
    pub n_rp: u32,
    pub n_ras: u32,
    pub n_rc: u32,
    pub n_wr: u32,
    pub n_rtpl: u32,
    pub n_cwl: u32,
    pub n_ccds: u32,
    pub n_ccdl: u32,
    pub n_wtrs: u32,
    pub n_wtrl: u32,
}

impl HBM2Timing {
    pub const HBM2_1600MBPS: HBM2Timing = HBM2Timing {
        rate: 1600,
        n_bl: 2,
        n_cl: 10,
        n_rcdrd: 10,
        n_rcdwr: 8,
        n_rp: 10,
        n_ras: 24,
        n_rc: 34,
        n_wr: 12,
        n_rtpl: 4,
        n_cwl: 4,
        n_ccds: 2,
        n_ccdl: 4,
        n_wtrs: 5,
        n_wtrl: 6,
    };
    pub const HBM2_2000MBPS: HBM2Timing = HBM2Timing {
        rate: 2000,
        n_bl: 2,
        n_cl: 14,
        n_rcdrd: 14,
        n_rcdwr: 12,
        n_rp: 14,
        n_ras: 34,
        n_rc: 48,
        n_wr: 16,
        n_rtpl: 5,
        n_cwl: 5,
        n_ccds: 2,
        n_ccdl: 4,
        n_wtrs: 6,
        n_wtrl: 8,
    };
    pub const HBM2_2400MBPS: HBM2Timing = HBM2Timing {
        rate: 2400,
        n_bl: 2,
        n_cl: 17,
        n_rcdrd: 17,
        n_rcdwr: 14,
        n_rp: 17,
        n_ras: 40,
        n_rc: 57,
        n_wr: 19,
        n_rtpl: 6,
        n_cwl: 6,
        n_ccds: 2,
        n_ccdl: 4,
        n_wtrs: 8,
        n_wtrl: 10,
    };
}

#[derive(Debug, Clone)]
pub struct HBM2Org {
    pub dq: u32,
    pub pseudochannel: u32,
    pub bankgroup: u32,
    pub bank: u32,
    pub row: u32,
    pub column: u32,
}

impl HBM2Org {
    pub const HBM2_1GB: HBM2Org = HBM2Org {
        dq: 64,
        pseudochannel: 2,
        bankgroup: 4,
        bank: 4,
        row: 1 << 13,
        column: (1 << 5) << 2,
    };
    pub const HBM2_2GB: HBM2Org = HBM2Org {
        row: 1 << 14,
        ..Self::HBM2_1GB
    };
    pub const HBM2_4GB: HBM2Org = HBM2Org {
        row: 1 << 15,
        ..Self::HBM2_1GB
    };
    pub const HBM2_8GB: HBM2Org = HBM2Org {
        row: 1 << 16,
        ..Self::HBM2_1GB
    };

    pub const fn density_in_mb(&self) -> u32 {
        (((self.dq * self.bankgroup * self.bank) as u64 * self.row as u64 * self.column as u64)
            / 1024
            / 1024) as u32
    }
}

const _: () = {
    assert!(HBM2Org::HBM2_1GB.density_in_mb() == 1024);
};

#[derive(Debug, Clone)]
pub struct HBM2 {
    pub timing: HBM2Timing,
    pub org: HBM2Org,
}

impl serde::Serialize for HBM2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.resolve_all().serialize(serializer)
    }
}

impl HBM2 {
    fn ns_to_cycle(&self, ns: u32) -> u32 {
        (ns * self.timing.rate).div_ceil(2000)
    }

    fn resolve_n_rfc(&self) -> u32 {
        let t_rfc = match self.org.density_in_mb() {
            ..=1024 => 110,
            ..=2048 => 160,
            ..=4096 => 260,
            ..=8192 => 350,
            _ => 450,
        };

        self.ns_to_cycle(t_rfc)
    }

    fn resolve_n_rfcsb(&self) -> u32 {
        let t_rfcsb = match self.org.density_in_mb() {
            ..=8192 => 160,
            _ => 200,
        };

        self.ns_to_cycle(t_rfcsb)
    }

    fn resolve_all(&self) -> serde_json::Value {
        let HBM2Timing {
            rate,
            n_bl,
            n_cl,
            n_rcdrd,
            n_rcdwr,
            n_rp,
            n_ras,
            n_rc,
            n_wr,
            n_rtpl,
            n_cwl,
            n_ccds,
            n_ccdl,
            n_wtrs,
            n_wtrl,
        } = self.timing;

        let n_rrds = max(4, self.ns_to_cycle(4));
        let n_rrdl = max(4, self.ns_to_cycle(4));
        let n_faw = max(8, self.ns_to_cycle(15));
        let n_rfc = self.resolve_n_rfc();
        let n_rfcsb = self.resolve_n_rfcsb();
        let n_rrefd = max(4, self.ns_to_cycle(8));
        let n_refi = self.ns_to_cycle(3900);
        let n_refisb = self.ns_to_cycle(
            3900u32.div_ceil(self.org.pseudochannel * self.org.bankgroup * self.org.bank),
        );

        let tck_ps = 2e6 / rate as f32;

        let timing_params = serde_json::json!([
            rate, n_bl, n_cl, n_rcdrd, n_rcdwr, n_rp, n_ras, n_rc, n_wr, n_rtpl, n_cwl, n_ccds,
            n_ccdl, n_rrds, n_rrdl, n_wtrs, n_wtrl, n_faw, n_rfc, n_rfcsb, n_rrefd, n_refi,
            n_refisb, tck_ps
        ]);

        const PSEUDOCHANNEL: u32 = 1;
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
        const REF_PB: u32 = 8;

        let timing_constraints = serde_json::json!([
            [PSEUDOCHANNEL, [RD, RDA], [RD, RDA], n_bl],
            [PSEUDOCHANNEL, [WR, WRA], [WR, WRA], n_bl],
            [PSEUDOCHANNEL, [RD, RDA], [RD, RDA], n_ccds],
            [PSEUDOCHANNEL, [WR, WRA], [WR, WRA], n_ccds],
            [
                PSEUDOCHANNEL,
                [RD, RDA],
                [WR, WRA],
                (n_cl + n_bl + 2).saturating_sub(n_cwl)
            ],
            [PSEUDOCHANNEL, [WR, WRA], [RD, RDA], n_cwl + n_bl + n_wtrs],
            [PSEUDOCHANNEL, [RD], [PRE_AB], n_rtpl],
            [PSEUDOCHANNEL, [WR], [PRE_AB], n_cwl + n_bl + n_wr],
            [PSEUDOCHANNEL, [ACT], [ACT], n_rrds],
            [PSEUDOCHANNEL, [ACT], [ACT], n_faw, 4],
            [PSEUDOCHANNEL, [ACT], [PRE_AB], n_ras],
            [PSEUDOCHANNEL, [PRE_AB], [ACT], n_rp],
            [PSEUDOCHANNEL, [ACT], [REF_AB], n_rc],
            [PSEUDOCHANNEL, [PRE_PB, PRE_AB], [REF_AB], n_rp],
            [PSEUDOCHANNEL, [RDA], [REF_AB], n_rp + n_rtpl],
            [PSEUDOCHANNEL, [WRA], [REF_AB], n_cwl + n_bl + n_wr + n_rp],
            [PSEUDOCHANNEL, [REF_AB], [ACT, PRE_AB], n_rfc],
            [PSEUDOCHANNEL, [REF_PB], [ACT], n_rrefd],
            [PSEUDOCHANNEL, [ACT], [REF_PB], n_rrds],
            [BANKGROUP, [RD, RDA], [RD, RDA], n_ccdl],
            [BANKGROUP, [WR, WRA], [WR, WRA], n_ccdl],
            [BANKGROUP, [WR, WRA], [RD, RDA], n_cwl + n_bl + n_wtrl],
            [BANKGROUP, [ACT], [ACT], n_rrdl],
            [BANK, [ACT], [ACT], n_rc],
            [BANK, [ACT], [RD, RDA], n_rcdrd],
            [BANK, [ACT], [WR, WRA], n_rcdwr],
            [BANK, [ACT], [PRE_PB], n_ras],
            [BANK, [PRE_PB], [ACT], n_rp],
            [BANK, [RD], [PRE_PB], n_rtpl],
            [BANK, [WR], [PRE_PB], n_cwl + n_bl + n_wr],
            [BANK, [RDA], [ACT], n_rtpl + n_rp],
            [BANK, [WRA], [ACT], n_cwl + n_bl + n_wr + n_rp],
            [BANK, [REF_PB], [ACT], n_rfcsb],
            [BANK, [ACT], [REF_PB], n_rc],
            [BANK, [PRE_PB], [REF_PB], n_rp],
        ]);

        serde_json::json!({
            "channel_width": 64,
            "org": {
                "dq": self.org.dq,
                "count": [1, self.org.pseudochannel, self.org.bankgroup, self.org.bank, self.org.row, self.org.column],
            },
            "timing": timing_params,
            "read_latency": n_cl + n_bl,
            "timing_constraints": timing_constraints,
        })
    }
}
