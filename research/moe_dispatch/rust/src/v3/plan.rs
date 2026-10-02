//! Executable, output-address-preserving tile plans. All bytes are 32-byte aligned.
use super::{Cfg, Expert, align, ceil};
#[derive(Clone, Debug)]
pub struct TileSpec {
    pub bytes: usize,
    pub main_bytes: usize,
    pub kv: usize,
    pub ranks: usize,
    pub projection: usize,
    pub kseg: usize,
    pub n: usize,
    pub kind: Kind,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Main,
    Prepass,
    Tail,
    Padding,
}
#[derive(Clone, Debug)]
pub struct Group {
    pub first_tile: usize,
    pub tile_count: usize,
    pub kind: Kind,
    pub projection: usize,
    pub kseg: usize,
    pub kv: usize,
    pub input_width: usize,
    pub passes: usize,
    pub accum_key: usize,
}
#[derive(Clone, Debug)]
pub enum Action {
    Group(Group),
    Vector {
        elements: usize,
        z_col: usize,
        z_cols: usize,
    },
    Drain,
    UStore {projection:usize,rank_start:usize,rank_cols:usize},
    UAccumulate {rank_start:usize,rank_cols:usize,k_segment:usize},
    Combine {
        col: usize,
        cols: usize,
        partial:bool,
        k_segment:usize,
    },
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub tiles: Vec<TileSpec>,
    pub actions: Vec<Action>,
    pub bytes: usize,
    pub main_tiles: usize,
    pub prepass_tiles: usize,
    pub short_tiles: usize,
    pub dataflow: String,
    pub z_mode: String,
    pub ranks: [usize; 3],
    pub baseline_bytes: usize,
    pub cross_core_bytes: usize,
    pub helper_split: bool,
    pub padding_bytes: usize,
}
fn factor(red: usize, n: usize, f: &str) -> usize {
    match f {
        "bf16" => red * n * 2,
        "mxint8" | "mxint8s" => red * n + n * ceil(red, 32),
        "mxint4" => ceil(red * n, 2) + n * ceil(red, 32),
        _ => 0,
    }
}
pub fn main_bytes(k: usize, bits: usize) -> usize {
    if bits == 16 {
        4 * k * 2
    } else {
        align(ceil(4 * k * bits, 8) + 4 * ceil(k, 32), 32)
    }
}
impl Plan {
    fn tile(&mut self, spec: TileSpec) -> usize {
        let i = self.tiles.len();
        self.bytes += spec.bytes;
        match spec.kind {
            Kind::Main => self.main_tiles += 1,
            Kind::Prepass => self.prepass_tiles += 1,
            Kind::Tail => self.short_tiles += 1,
            Kind::Padding => {}
        };
        self.tiles.push(spec);
        i
    }
    fn group(
        &mut self,
        tiles: Vec<usize>,
        kind: Kind,
        p: usize,
        s: usize,
        k: usize,
        iw: usize,
        passes: usize,
        key: usize,
    ) {
        if !tiles.is_empty() {
            assert!(tiles.windows(2).all(|w| w[1] == w[0] + 1));
            if self.helper_split && (kind == Kind::Prepass || kind == Kind::Tail) {
                for tile in tiles {
                    self.actions.push(Action::Group(Group {
                        first_tile: tile,
                        tile_count: 1,
                        kind,
                        projection: p,
                        kseg: s,
                        kv: k,
                        input_width: iw,
                        passes,
                        accum_key: key,
                    }));
                    if kind==Kind::Prepass&&p==4 {self.actions.push(Action::UAccumulate{rank_start:self.tiles[tile].n,rank_cols:4.min(self.ranks[2].saturating_sub(self.tiles[tile].n)),k_segment:s});}
                }
            } else {
                self.actions.push(Action::Group(Group {
                    first_tile: tiles[0],
                    tile_count: tiles.len(),
                    kind,
                    projection: p,
                    kseg: s,
                    kv: k,
                    input_width: iw,
                    passes,
                    accum_key: key,
                }));
                if kind==Kind::Prepass&&p==4 {self.actions.push(Action::UAccumulate{rank_start:self.tiles[tiles[0]].n,rank_cols:(tiles.len()*4).min(self.ranks[2].saturating_sub(self.tiles[tiles[0]].n)),k_segment:s});}
            }
        }
    }
    pub fn append_padding(&mut self, bytes: usize) {
        assert_eq!(bytes % 32, 0);
        self.padding_bytes = bytes;
        let mut left = bytes;
        while left > 0 {
            let size = left.min(4096);
            let tile = self.tile(TileSpec {
                bytes: size,
                main_bytes: 0,
                kv: 0,
                ranks: 0,
                projection: 9,
                kseg: 0,
                n: 0,
                kind: Kind::Padding,
            });
            self.group(vec![tile], Kind::Padding, 9, 0, 0, 0, 1, 0);
            left -= size;
        }
    }
    fn a_segment(&mut self, cfg: &Cfg, p: usize, k: usize, nrank: usize, s: usize, g: usize) {
        if nrank == 0 {
            return;
        }
        let kv = (k - s * 512).min(512);
        let mut items = vec![];
        for n in (0..nrank).step_by(4) {
            let b = align(factor(kv, 4, &cfg.factor_a), 32);
            items.push(self.tile(TileSpec {
                bytes: b,
                main_bytes: b,
                kv,
                ranks: 0,
                projection: p,
                kseg: s,
                n,
                kind: Kind::Prepass,
            }));
            if items.len() == g {
                self.group(
                    std::mem::take(&mut items),
                    Kind::Prepass,
                    p,
                    s,
                    kv,
                    k,
                    if cfg.factor_a == "mxint8s" { 2 } else { 1 },
                    p * 1_000_000 + n / 4,
                );
            }
        }
        self.group(
            items,
            Kind::Prepass,
            p,
            s,
            kv,
            k,
            if cfg.factor_a == "mxint8s" { 2 } else { 1 },
            p * 1_000_000,
        );
    }
    fn a_all(&mut self,cfg:&Cfg,p:usize,k:usize,nrank:usize,g:usize) {
        if self.dataflow=="ws_group"&&p==3 {
            for start in (0..nrank).step_by(g*4) {
                let end=(start+g*4).min(nrank);
                for seg in 0..ceil(k,512) {
                    let kv=(k-seg*512).min(512);let mut ids=vec![];
                    for col in (start..end).step_by(4) {let bytes=align(factor(kv,4,&cfg.factor_a),32);ids.push(self.tile(TileSpec{bytes,main_bytes:bytes,kv,ranks:0,projection:p,kseg:seg,n:col,kind:Kind::Prepass}));}
                    self.group(ids,Kind::Prepass,p,seg,kv,k,if cfg.factor_a=="mxint8s"{2}else{1},p*1_000_000+start/4);
                }
                self.actions.push(Action::Drain);self.actions.push(Action::UStore{projection:p,rank_start:start,rank_cols:end-start});
            }
        } else {
            for seg in 0..ceil(k,512){self.a_segment(cfg,p,k,nrank,seg,g);}
            self.actions.push(Action::Drain);self.actions.push(Action::UStore{projection:p,rank_start:0,rank_cols:nrank});
        }
    }
    fn main_tile(
        &mut self,
        cfg: &Cfg,
        p: usize,
        k: usize,
        n: usize,
        s: usize,
        rank: usize,
        late: bool,
    ) -> usize {
        let original_s = ceil(k, 512);
        let effective_k = if cfg.comp == "kext" { k + rank } else { k };
        let lo = s * 512;
        let hi = (lo + 512).min(effective_k);
        let kw = k.saturating_sub(lo).min(hi - lo);
        let kb = if cfg.comp == "kext" {
            hi.saturating_sub(k.max(lo)).min(rank)
        } else {
            0
        };
        let main = if kw > 0 { main_bytes(kw, cfg.bits) } else { 0 };
        let capacity = if late {
            cfg.rank_lanes
        } else {
            cfg.rank_lanes * original_s
        };
        let fused = if cfg.comp == "lanes" {
            rank.min(capacity)
        } else {
            0
        };
        let nr = if cfg.comp == "kext" {
            kb
        } else if late {
            if s + 1 == original_s { fused } else { 0 }
        } else {
            fused / original_s + usize::from(s < fused % original_s)
        };
        let slack = if cfg.slack_pack && cfg.comp == "lanes" && !late && s + 1 == original_s {
            rank.saturating_sub(fused).min(original_s * 512 - k)
        } else {
            0
        };
        let b = align(main + factor(nr + slack, 4, &cfg.factor_b), 32);
        self.tile(TileSpec {
            bytes: b,
            main_bytes: main,
            kv: kw,
            ranks: nr + slack,
            projection: p,
            kseg: s,
            n,
            kind: Kind::Main,
        })
    }
    fn tail(&mut self,cfg:&Cfg,p:usize,k:usize,n:usize,r:usize,g:usize,late:bool){self.tail_range(cfg,p,k,0,n,r,g,late);}
    fn tail_range(&mut self, cfg: &Cfg, p: usize, k: usize, nstart:usize,n: usize, r: usize, g: usize, late: bool) {
        let cap = if late {
            cfg.rank_lanes
        } else {
            cfg.rank_lanes * ceil(k, 512)
                + if cfg.slack_pack {
                    ceil(k, 512) * 512 - k
                } else {
                    0
                }
        };
        let tail = if cfg.comp == "separate" || cfg.comp == "offload" {
            r
        } else {
            r.saturating_sub(cap)
        };
        if tail == 0 || cfg.comp == "kext" || cfg.comp == "none" {
            return;
        }
        let step = if cfg.comp == "lanes" {
            cfg.rank_lanes.max(1)
        } else {
            512
        };
        for start in (nstart..n).step_by(g * 4) {
            for seg in 0..ceil(tail, step) {
                let nr = (tail - seg * step).min(step);
                let mut ids = vec![];
                for col in (start..(start + g * 4).min(n)).step_by(4) {
                    let b = align(factor(nr, 4, &cfg.factor_b), 32);
                    ids.push(self.tile(TileSpec {
                        bytes: b,
                        main_bytes: b,
                        kv: nr,
                        ranks: nr,
                        projection: p,
                        kseg: ceil(k, 512) + seg,
                        n: col,
                        kind: Kind::Tail,
                    }));
                }
                self.group(
                    ids,
                    Kind::Tail,
                    p,
                    ceil(k, 512) + seg,
                    nr,
                    r,
                    1,
                    p * 1_000_000 + start / 4,
                );
                if p==2&&late {self.actions.push(Action::Combine{col:start,cols:(start+g*4).min(n)-start,partial:true,k_segment:ceil(k,512)+seg});}
            }
        }
    }
}
pub fn build(e: &Expert, cfg: &Cfg, core: usize, flow: &str, g: usize) -> Result<Plan, String> {
    let r = if cfg.precision == "P0" || cfg.comp == "none" {
        [0, 0, 0]
    } else {
        cfg.ranks(e.shared)
    };
    let z = if cfg.z_mode == "auto" {
        if e.m > 64 { "streamed" } else { "full" }
    } else {
        &cfg.z_mode
    };
    if flow == "is_stream" && e.m > if cfg.flows[core]=="is_stream"{4}else{cfg.lanes[core]} {
        return Err("streaming task exceeds Me=4 hard bound".into());
    }
    let mut p = Plan {
        tiles: vec![],
        actions: vec![],
        bytes: 0,
        main_tiles: 0,
        prepass_tiles: 0,
        short_tiles: 0,
        dataflow: flow.into(),
        z_mode: z.into(),
        ranks: r,
        baseline_bytes: 6 * e.h * e.f,
        cross_core_bytes: 0,
        helper_split: cfg.comp == "offload",
        padding_bytes: 0,
    };
    if cfg.comp == "offload" {
        p.cross_core_bytes = e.m * (2 * e.h + 2 * e.f + 4 * (2 * e.f + e.h));
    }
    let g = g.max(1).min(cfg.wor_slots(core).max(1));
    let a_slots=if cfg.precision=="P2"{ceil(align(factor(512,4,&cfg.factor_a),32),cfg.wslot_bytes())}else{1};
    let ag=g.min((cfg.wor_slots(core)/a_slots.max(1)).max(1));
    let groups_gu = (g / 2).max(1);
    if r[0] + r[1] > 0 {
        p.a_all(cfg, 3, e.h, r[0] + r[1], ag);
    }
    if flow == "is_stream" && z!="streamed" {
        for s in 0..ceil(e.h, 512) {
            for proj in 0..2 {
                for start in (0..e.f).step_by(g * 4) {
                    let ids = (start..(start + g * 4).min(e.f))
                        .step_by(4)
                        .map(|n| p.main_tile(cfg, proj, e.h, n, s, r[proj], false))
                        .collect();
                    p.group(
                        ids,
                        Kind::Main,
                        proj,
                        s,
                        (e.h - s * 512).min(512),
                        e.h,
                        1,
                        proj * 1_000_000 + start / 4,
                    );
                }
            }
        }
        p.tail(cfg, 0, e.h, e.f, r[0], g, false);
        p.tail(cfg, 1, e.h, e.f, r[1], g, false);
        p.actions.push(Action::Vector {
            elements: e.m * e.f,
            z_col: 0,
            z_cols: e.f,
        });
        if r[2] > 0 {
            p.a_all(cfg, 4, e.f, r[2], ag);
        }
        for s in 0..ceil(if cfg.comp == "kext" { e.f + r[2] } else { e.f }, 512) {
            for start in (0..e.h).step_by(g * 4) {
                let ids = (start..(start + g * 4).min(e.h))
                    .step_by(4)
                    .map(|n| p.main_tile(cfg, 2, e.f, n, s, r[2], false))
                    .collect();
                p.group(
                    ids,
                    Kind::Main,
                    2,
                    s,
                    e.f.saturating_sub(s * 512).min(512),
                    e.f,
                    1,
                    2_000_000 + start / 4,
                );
            }
        }
        p.tail(cfg, 2, e.f, e.h, r[2], g, false);
        p.actions.push(Action::Drain);
        p.actions.push(Action::Combine { col: 0, cols: e.h,partial:false,k_segment:usize::MAX });
    } else {
        let streamed = z == "streamed";
        let segments = ceil(e.f, 512);
        for zseg in 0..segments {
            let zlo = zseg * 512;
            let zhi = (zlo + 512).min(e.f);
            for start in (zlo..zhi).step_by(groups_gu * 4) {
                for s in 0..ceil(
                    if cfg.comp == "kext" {
                        e.h + r[0].max(r[1])
                    } else {
                        e.h
                    },
                    512,
                ) {
                    let mut ids = vec![];
                    for proj in 0..2 {
                        for n in (start..(start + groups_gu * 4).min(zhi)).step_by(4) {
                            if s < ceil(e.h + if cfg.comp == "kext" { r[proj] } else { 0 }, 512) {
                                ids.push(p.main_tile(cfg, proj, e.h, n, s, r[proj], false));
                            }
                        }
                    }
                    p.group(
                        ids,
                        Kind::Main,
                        0,
                        s,
                        e.h.saturating_sub(s * 512).min(512),
                        e.h,
                        1,
                        start / 4,
                    );
                }
                // Short passes on Gate/Up use the same output-column group before SiLU.
                for proj in 0..2 {
                    let cap = cfg.rank_lanes * ceil(e.h, 512);
                    let tail = if cfg.comp == "separate" || cfg.comp == "offload" {
                        r[proj]
                    } else if cfg.comp == "lanes" {
                        r[proj].saturating_sub(cap)
                    } else {
                        0
                    };
                    if tail > 0 {
                        let step = if cfg.comp == "lanes" {
                            cfg.rank_lanes.max(1)
                        } else {
                            512
                        };
                        for ts in 0..ceil(tail, step) {
                            let nr = (tail - ts * step).min(step);
                            let mut ids = vec![];
                            for n in (start..(start + groups_gu * 4).min(zhi)).step_by(4) {
                                let b = align(factor(nr, 4, &cfg.factor_b), 32);
                                ids.push(p.tile(TileSpec {
                                    bytes: b,
                                    main_bytes: b,
                                    kv: nr,
                                    ranks: nr,
                                    projection: proj,
                                    kseg: ceil(e.h, 512) + ts,
                                    n,
                                    kind: Kind::Tail,
                                }));
                            }
                            p.group(
                                ids,
                                Kind::Tail,
                                proj,
                                ceil(e.h, 512) + ts,
                                nr,
                                r[proj],
                                1,
                                proj * 1_000_000 + start / 4,
                            );
                        }
                    }
                }
                p.actions.push(Action::Vector {
                    elements: e.m * ((start + groups_gu * 4).min(zhi) - start),
                    z_col: start,
                    z_cols: (start + groups_gu * 4).min(zhi) - start,
                });
            }
            if r[2] > 0 {
                p.a_segment(cfg, 4, e.f, r[2], zseg, ag);
                p.actions.push(Action::Drain);
                if zseg+1==segments{p.actions.push(Action::UStore{projection:4,rank_start:0,rank_cols:r[2]});}
            }
            if streamed {
                for start in (0..e.h).step_by(g * 4) {
                    let ids = (start..(start + g * 4).min(e.h))
                        .step_by(4)
                        .map(|n| p.main_tile(cfg, 2, e.f, n, zseg, r[2], true))
                        .collect();
                    p.group(
                        ids,
                        Kind::Main,
                        2,
                        zseg,
                        (e.f - zseg * 512).min(512),
                        e.f,
                        1,
                        2_000_000 + start / 4,
                    );
                    p.actions.push(Action::Combine{col:start,cols:(start+g*4).min(e.h)-start,partial:true,k_segment:zseg});
                }
            }
        }
        if !streamed {
            p.actions.push(Action::Drain);
            for start in (0..e.h).step_by(g * 4) {
                for s in 0..ceil(if cfg.comp == "kext" { e.f + r[2] } else { e.f }, 512) {
                    let ids = (start..(start + g * 4).min(e.h))
                        .step_by(4)
                        .map(|n| p.main_tile(cfg, 2, e.f, n, s, r[2], false))
                        .collect();
                    p.group(
                        ids,
                        Kind::Main,
                        2,
                        s,
                        e.f.saturating_sub(s * 512).min(512),
                        e.f,
                        1,
                        2_000_000 + start / 4,
                    );
                }
                p.tail_range(cfg,2,e.f,start,(start+g*4).min(e.h),r[2],g,false);
                p.actions.push(Action::Combine{col:start,cols:(start+g*4).min(e.h)-start,partial:false,k_segment:usize::MAX});
            }
        } else {
            if cfg.comp=="kext" {
                for seg in segments..ceil(e.f+r[2],512){for start in (0..e.h).step_by(g*4){let ids=(start..(start+g*4).min(e.h)).step_by(4).map(|n|p.main_tile(cfg,2,e.f,n,seg,r[2],true)).collect();p.group(ids,Kind::Main,2,seg,0,e.f,1,2_000_000+start/4);p.actions.push(Action::Combine{col:start,cols:(start+g*4).min(e.h)-start,partial:true,k_segment:seg});}}
            }
            p.tail(cfg,2,e.f,e.h,r[2],g,true);
        }
    }
    if cfg.comp == "offload" && cfg.lanes.len() < 2 {
        return Err("offload requires a second core".into());
    }
    let _ = core;
    Ok(p)
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn exact_default_bytes() {
        let c = Cfg::parse(&json!({})).unwrap();
        let e = Expert {
            id: 0,
            shared: false,
            m: 2,
            h: 2048,
            f: 1408,
            tokens: vec![0, 1],
        };
        let p = build(&e, &c, 0, "ws_group", 8).unwrap();
        assert_eq!(p.bytes, 4_970_112);
        assert_eq!(p.main_tiles, 4352);
        assert_eq!(p.prepass_tiles, 82);
        let sh = Expert {
            shared: true,
            f: 2816,
            ..e
        };
        let p = build(&sh, &c, 0, "ws_group", 8).unwrap();
        assert_eq!(p.bytes, 9_889_920);
    }
    #[test]
    fn is_hard_bound() {
        let c = Cfg::parse(&json!({})).unwrap();
        let e = Expert {
            id: 0,
            shared: false,
            m: 5,
            h: 512,
            f: 16,
            tokens: vec![],
        };
        assert!(build(&e, &c, 0, "is_stream", 1).is_err());
    }
}
