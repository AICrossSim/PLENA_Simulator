//! Optional functional arithmetic driven by the same issued tile stream as timing.
//! This does not add modeled cycles. Payload values are unpacked MX/BF16 values,
//! not original unquantized weights; absent payload means no functional work.
use super::plan::{Kind, TileSpec};
use super::{Cfg, ceil};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
type Mat = Vec<Vec<f32>>;
fn bf(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn matrix(v: &Value) -> Result<Mat, String> {
    v.as_array()
        .ok_or("numeric matrix missing")?
        .iter()
        .map(|r| {
            r.as_array()
                .ok_or("numeric row missing")?
                .iter()
                .map(|x| {
                    x.as_f64()
                        .map(|z| z as f32)
                        .ok_or_else(|| "numeric value invalid".to_string())
                })
                .collect()
        })
        .collect()
}
struct Factor {
    a: Mat,
    b: Mat,
    hi: Option<Mat>,
    lo: Option<Mat>,
}
struct Expert {
    id: i64,
    x: Mat,
    w: [Mat; 3],
    factors: [Factor; 3],
    gate: Vec<f32>,
    m: usize,
    h: usize,
    f: usize,
    u: [Mat; 3],
    u_count: [Vec<Vec<usize>>; 3],
    out: [Mat; 3],
    seen: BTreeSet<(usize, usize, usize, usize)>,
    rank_seen: BTreeSet<(usize, usize, usize, usize)>,
    z: Mat,
    z_ready: Vec<bool>,
    split_pass: BTreeMap<(usize, usize, usize, usize), usize>,
    corrected: bool,
    issues: u64,
    pending: Mat,
    // Preserve actual issue order and valid row ranges for independent gold replay.
    pending_sources: Vec<(usize, usize, bool, Vec<usize>, usize, usize)>,
}
impl Expert {
    fn new(v: &Value) -> Result<Self, String> {
        let mut x = matrix(&v["x"])?;
        let m = x.len();
        let h = x.first().ok_or("numeric empty X")?.len();
        for row in &mut x {
            if row.len() != h {
                return Err("numeric X shape".into());
            }
            for val in row {
                *val = bf(*val);
            }
        }
        let w = [
            matrix(&v["weights"]["g"])?,
            matrix(&v["weights"]["u"])?,
            matrix(&v["weights"]["d"])?,
        ];
        let f = w[0].len();
        if w[1].len() != f
            || w[2].len() != h
            || w[0].iter().chain(&w[1]).any(|r| r.len() != h)
            || w[2].iter().any(|r| r.len() != f)
        {
            return Err("numeric FFN dimensions disagree".into());
        }
        let mut fac = Vec::new();
        for (p, name) in ["g", "u", "d"].iter().enumerate() {
            let q = &v["factors"][name];
            let k = if p == 2 { f } else { h };
            let n = if p == 2 { h } else { f };
            let a = if q["a"].is_array() {
                matrix(&q["a"])?
            } else {
                vec![vec![]; k]
            };
            let r = a.first().map_or(0, Vec::len);
            let b = if q["b"].is_array() {
                matrix(&q["b"])?
            } else {
                vec![]
            };
            if a.len() != k
                || a.iter().any(|z| z.len() != r)
                || b.len() != r
                || b.iter().any(|z| z.len() != n)
            {
                return Err("numeric A/B dimensions disagree".into());
            }
            let hi = q.get("a_hi").map(matrix).transpose()?;
            let lo = q.get("a_lo").map(matrix).transpose()?;
            fac.push(Factor { a, b, hi, lo });
        }
        let factors: [Factor; 3] = fac.try_into().map_err(|_| "numeric factor count")?;
        let gate = (0..m)
            .map(|i| {
                v["gate"]
                    .get(i)
                    .and_then(Value::as_f64)
                    .or_else(|| v["gate"].as_f64())
                    .unwrap_or(1.) as f32
            })
            .collect();
        let u = std::array::from_fn(|p| vec![vec![0.; factors[p].b.len()]; m]);
        let u_count = std::array::from_fn(|p| vec![vec![0; factors[p].b.len()]; m]);
        let out = std::array::from_fn(|p| vec![vec![0.; if p == 2 { h } else { f }]; m]);
        Ok(Self {
            id: v["expert_id"].as_i64().unwrap_or(-1),
            x,
            w,
            factors,
            gate,
            m,
            h,
            f,
            u,
            u_count,
            out,
            seen: BTreeSet::new(),
            rank_seen: BTreeSet::new(),
            z: vec![vec![0.; f]; m],
            z_ready: vec![false; f],
            split_pass: BTreeMap::new(),
            corrected: true,
            issues: 0,
            pending: vec![vec![0.; h]; m],
            pending_sources: Vec::new(),
        })
    }
    fn issue(
        &mut self,
        s: &TileSpec,
        start: usize,
        rows: usize,
        kind: Kind,
        cfg: &Cfg,
        z_mode: &str,
    ) -> Result<(), String> {
        if kind == Kind::Padding {
            return Ok(());
        }
        if start + rows > self.m {
            return Err("numeric issue rows exceed expert input".into());
        }
        self.corrected = cfg.comp != "none" && cfg.precision != "P0";
        self.issues += 1;
        if kind == Kind::Prepass {
            let split = cfg.factor_a == "mxint8s";
            let key = (s.projection, s.kseg, s.n, start);
            let phase = *self.split_pass.entry(key).or_default();
            *self.split_pass.get_mut(&key).unwrap() += 1;
            for row in start..start + rows {
                for n in s.n..s.n + 4 {
                    let (p, j) = if s.projection == 4 {
                        (2, n)
                    } else if n < self.factors[0].b.len() {
                        (0, n)
                    } else {
                        (1, n - self.factors[0].b.len())
                    };
                    if j >= self.factors[p].b.len() {
                        continue;
                    }
                    let k = if p == 2 { self.f } else { self.h };
                    let lo = s.kseg * 512;
                    let hi = (lo + s.kv).min(k);
                    if p == 2 && (lo..hi).any(|q| !self.z_ready[q]) {
                        return Err("numeric U_d consumes Z before produced".into());
                    }
                    let a = if split {
                        if phase % 2 == 0 {
                            self.factors[p].hi.as_ref()
                        } else {
                            self.factors[p].lo.as_ref()
                        }
                        .ok_or("MXINT8-split numeric replay needs actual a_hi and a_lo payload")?
                    } else {
                        &self.factors[p].a
                    };
                    let mut sum = 0.0f32;
                    for q in lo..hi {
                        let x = if p == 2 {
                            self.z[row][q]
                        } else {
                            self.x[row][q]
                        };
                        sum += x * a[q][j];
                    }
                    self.u[p][row][j] += sum;
                    self.u_count[p][row][j] += 1;
                }
            }
            return Ok(());
        }
        if s.projection > 2 {
            return Err("numeric output projection invalid".into());
        }
        let p = s.projection;
        let k = if p == 2 { self.f } else { self.h };
        let n = self.w[p].len();
        let segments = ceil(k, 512);
        let r = self.factors[p].b.len();
        let streamed = p == 2 && z_mode == "streamed";
        let rank_ids: Vec<usize> = if !self.corrected {
            vec![]
        } else if kind == Kind::Tail {
            let cap = if cfg.comp == "lanes" {
                if streamed {
                    cfg.rank_lanes
                } else {
                    cfg.rank_lanes * segments
                        + if cfg.slack_pack {
                            segments * 512 - k
                        } else {
                            0
                        }
                }
            } else {
                0
            };
            let stride = if cfg.comp == "lanes" {
                cfg.rank_lanes.max(1)
            } else {
                512
            };
            let first = cap + s.kseg.saturating_sub(segments) * stride;
            (first..(first + s.ranks).min(r)).collect()
        } else if cfg.comp == "kext" {
            let first = (s.kseg * 512).saturating_sub(k);
            (first..(first + s.ranks).min(r)).collect()
        } else if cfg.comp == "lanes" {
            if streamed {
                if s.kseg + 1 == segments {
                    (0..s.ranks.min(r)).collect()
                } else {
                    vec![]
                }
            } else {
                let fused = r.min(cfg.rank_lanes * segments);
                let mut ids: Vec<_> = (0..fused).filter(|j| j % segments == s.kseg).collect();
                if cfg.slack_pack && s.kseg + 1 == segments {
                    ids.extend(fused..r.min(fused + segments * 512 - k));
                }
                ids
            }
        } else {
            vec![]
        };
        for row in start..start + rows {
            for col in s.n..(s.n + 4).min(n) {
                let mut main = 0.0f32;
                if kind == Kind::Main {
                    if !self.seen.insert((p, row, col, s.kseg)) {
                        return Err("numeric main output segment issued twice".into());
                    }
                    let lo = s.kseg * 512;
                    let hi = (lo + s.kv).min(k);
                    if p == 2 && (lo..hi).any(|q| !self.z_ready[q]) {
                        return Err("numeric Down consumes unproduced Z".into());
                    }
                    for q in lo..hi {
                        main += if p == 2 {
                            self.z[row][q]
                        } else {
                            self.x[row][q]
                        } * self.w[p][col][q];
                    }
                }
                let mut rank = 0.0f32;
                for &j in &rank_ids {
                    if self.u_count[p][row][j]
                        < segments * (if cfg.factor_a == "mxint8s" { 2 } else { 1 })
                    {
                        return Err("numeric rank lane consumes incomplete U".into());
                    }
                    if !self.rank_seen.insert((p, row, col, j)) {
                        return Err("numeric correction rank applied twice".into());
                    }
                    rank += bf(self.u[p][row][j]) * self.factors[p].b[j][col];
                }
                let delta=main+rank;
                self.out[p][row][col] += delta;
                if p==2 {self.pending[row][col]+=delta;}
            }
        }
        if p==2 {self.pending_sources.push((s.n,s.kseg,kind==Kind::Main,rank_ids,start,rows));}
        Ok(())
    }
    fn vector(&mut self, col: usize, cols: usize) -> Result<(), String> {
        for c in col..(col + cols).min(self.f) {
            for row in 0..self.m {
                for p in 0..2 {
                    if (0..ceil(self.h, 512)).any(|s| !self.seen.contains(&(p, row, c, s))) {
                        return Err("numeric SiLU consumes incomplete main projection".into());
                    }
                    if self.corrected
                        && (0..self.factors[p].b.len())
                            .any(|j| !self.rank_seen.contains(&(p, row, c, j)))
                    {
                        return Err("numeric SiLU precedes Gate/Up correction".into());
                    }
                }
                let g = self.out[0][row][c];
                self.z[row][c] = bf((g / (1. + (-g).exp())) * self.out[1][row][c] * self.gate[row]);
            }
            self.z_ready[c] = true;
        }
        Ok(())
    }
    fn report(&self) -> Value {
        json!({"expert_id":self.id,"output":self.out[2],"Z":self.z,"U_g":self.u[0].iter().map(|r|r.iter().map(|&x|bf(x)).collect::<Vec<_>>()).collect::<Vec<_>>(),"U_u":self.u[1].iter().map(|r|r.iter().map(|&x|bf(x)).collect::<Vec<_>>()).collect::<Vec<_>>(),"U_d":self.u[2].iter().map(|r|r.iter().map(|&x|bf(x)).collect::<Vec<_>>()).collect::<Vec<_>>(),"issued_tiles_with_numeric_work":self.issues,"all_Z_produced":self.z_ready.iter().all(|&x|x)})
    }
    fn finish(&self) -> Result<(), String> {
        if !self.z_ready.iter().all(|&ready| ready) {
            return Err(format!("numeric expert {} has incomplete Z", self.id));
        }
        for row in 0..self.m {
            for col in 0..self.h {
                if (0..ceil(self.f, 512)).any(|k| !self.seen.contains(&(2,row,col,k))) {
                    return Err(format!("numeric expert {} has incomplete Down",self.id));
                }
                if self.corrected && (0..self.factors[2].b.len()).any(|j| !self.rank_seen.contains(&(2,row,col,j))) {
                    return Err(format!("numeric expert {} has incomplete Down correction",self.id));
                }
                if self.pending[row][col] != 0.0 {
                    return Err(format!("numeric expert {} has uncombined partial output",self.id));
                }
            }
        }
        if !self.pending_sources.is_empty() {
            return Err(format!("numeric expert {} has unconsumed Down sources",self.id));
        }
        Ok(())
    }
}
pub(crate) struct NumState {
    experts: BTreeMap<i64, Expert>,
    combined: Option<Mat>,
    combine_events: Vec<Value>,
}
impl NumState {
    pub(crate) fn new(v: &Value) -> Result<Option<Self>, String> {
        if v.is_null() {
            return Ok(None);
        }
        let values: Vec<&Value> = if let Some(a) = v.get("experts").and_then(Value::as_array) {
            a.iter().collect()
        } else {
            vec![v]
        };
        let mut experts = BTreeMap::new();
        for x in values {
            let e = Expert::new(x)?;
            if experts.insert(e.id, e).is_some() {
                return Err("duplicate numeric expert payload".into());
            }
        }
        Ok(Some(Self {
            experts,
            combined: None,
            combine_events: Vec::new(),
        }))
    }
    pub(crate) fn on_issue(
        &mut self,
        id: i64,
        s: &TileSpec,
        m: usize,
        rows: usize,
        k: Kind,
        cfg: &Cfg,
        z_mode: &str,
    ) -> Result<(), String> {
        if let Some(e) = self.experts.get_mut(&id) {
            e.issue(s, m, rows, k, cfg, z_mode)
        } else {
            Err(format!("numeric payload absent for issued expert {id}"))
        }
    }
    pub(crate) fn on_vector(&mut self, id: i64, c: usize, n: usize) -> Result<(), String> {
        self.experts
            .get_mut(&id)
            .ok_or("numeric expert missing")?
            .vector(c, n)
    }
    pub(crate) fn on_combine(
        &mut self,
        id: i64,
        col: usize,
        cols: usize,
        tokens: &[usize],
        batch: usize,
        hidden: usize,
        partial: bool,
    ) -> Result<(), String> {
        let e = self
            .experts
            .get_mut(&id)
            .ok_or("numeric combine expert absent")?;
        if tokens.len() != e.m || e.h != hidden || tokens.iter().any(|&t| t >= batch) {
            return Err("numeric combine routing shape invalid".into());
        }
        let y = self
            .combined
            .get_or_insert_with(|| vec![vec![0.0; hidden]; batch]);
        if y.len() != batch || y[0].len() != hidden {
            return Err("numeric combine dimensions changed".into());
        }
        for (row, &t) in tokens.iter().enumerate() {
            for c in col..(col + cols).min(hidden) {
                if !partial && (0..ceil(e.f, 512)).any(|s| !e.seen.contains(&(2, row, c, s))) {
                    return Err("numeric combine before complete Down".into());
                }
                if !partial && e.corrected
                    && (0..e.factors[2].b.len()).any(|j| !e.rank_seen.contains(&(2, row, c, j)))
                {
                    return Err("numeric combine before Down correction".into());
                }
                y[t][c] += e.pending[row][c];
                e.pending[row][c]=0.;
            }
        }
        let sources:Vec<_>=e.pending_sources.iter().filter(|(n,_,_,_,_,_)|*n>=col&&*n<col+cols).map(|(n,s,main,ranks,start,rows)|json!({"n":n,"k_segment":s,"main":main,"ranks":ranks,"m_start":start,"valid_rows":rows})).collect();
        e.pending_sources.retain(|(n,_,_,_,_,_)|*n<col||*n>=col+cols);
        self.combine_events.push(json!({"expert_id":id,"col":col,"cols":cols,"tokens":tokens,"partial":partial,"sources":sources}));
        Ok(())
    }
    pub(crate) fn report(&self) -> Value {
        json!({"schema":"plena_v3_timed_functional_v1","scope":"arithmetic driven by actual timed main/prepass/tail and combine events; host arithmetic adds no modeled cycles","experts":self.experts.values().map(Expert::report).collect::<Vec<_>>(),"combined_output":self.combined,"combine_events":self.combine_events})
    }
    pub(crate) fn finish(&self) -> Result<(), String> {
        for expert in self.experts.values() { expert.finish()?; }
        Ok(())
    }
}
