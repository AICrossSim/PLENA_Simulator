//! Supply-first v3 cycle/event analytical model; not native HBM or RTL.
//! Every 32-byte request, physical pool bank, XOR transfer and array issue gates time.
mod memory;
mod plan;
mod replay;
#[cfg(test)]
mod tests_extra;
mod timed_numeric;
mod joint_policy;
use memory::{BankPort, Pool};
use plan::{Action, Kind, Plan};
pub use replay::replay;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
pub(crate) fn ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b.max(1))
}
pub(crate) fn align(a: usize, b: usize) -> usize {
    ceil(a, b) * b
}
fn n(v: &Value, k: &str, d: usize) -> usize {
    v.get(k)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .unwrap_or(d)
}
fn flag(v: &Value, k: &str, d: bool) -> bool {
    v.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn s(v: &Value, k: &str, d: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or(d).into()
}
fn vector(v: &Value, k: &str, d: Vec<usize>) -> Vec<usize> {
    match v.get(k) {
        Some(Value::Array(a)) => a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
        Some(Value::Number(x)) => vec![x.as_u64().unwrap_or(0) as usize; d.len()],
        _ => d,
    }
}
#[derive(Clone)]
pub(crate) struct Expert {
    pub id: i64,
    pub shared: bool,
    pub m: usize,
    pub h: usize,
    pub f: usize,
    pub tokens: Vec<usize>,
}
#[derive(Clone)]
pub(crate) struct Cfg {
    raw: Value,
    pub lanes: Vec<usize>,
    pub precision: String,
    pub bits: usize,
    pub rank_lanes: usize,
    pub comp: String,
    pub factor_a: String,
    pub factor_b: String,
    pub slack_pack: bool,
    pub z_mode: String,
    bw: usize,
    lat: u64,
    credits: usize,
    release: String,
    pool: usize,
    pool_banks: usize,
    ingress: usize,
    flows: Vec<String>,
    g: Vec<usize>,
    rd: Vec<usize>,
    xbw: Vec<usize>,
    accbw:Vec<usize>,
    rank_rd:Vec<usize>,
    rank_wr:Vec<usize>,
    dec: usize,
    vlen: usize,
    placement: String,
    contexts: usize,
    window: usize,
    max: u64,
    trace: bool,
    trace_limit: usize,
    ideal_hbm: bool,
    ideal_chip: bool,
    x_reuse: bool,
    w_reuse: bool,
    pipeline: bool,
    inline: bool,
    quota: bool,
    quota_policy:String,
    byte_pool: bool,
    control: bool,
    margin: usize,
    guard: bool,
    kappa: f64,
    m0: u64,
    rank_alloc: String,
    ranks_rt: [usize; 3],
    ranks_sh: [usize; 3],
}
impl Cfg {
    pub(crate) fn parse(v: &Value) -> Result<Self, String> {
        let lanes = vector(v, "lanes", vec![4, 2]);
        if lanes.is_empty() || lanes.len() > 2 || lanes.iter().any(|&m| m == 0) {
            return Err("v3 supports one or two positive-row cores".into());
        }
        let precision = s(v, "precision", "P2");
        if !["P0", "P1", "P2"].contains(&precision.as_str()) {
            return Err("precision must be P0/P1/P2".into());
        }
        if !["little","fixed_one_tile"].contains(&s(v,"quota_policy","little").as_str()){return Err("quota_policy must be little or fixed_one_tile".into());}
        let l = n(v, "rank_lanes", 8);
        let comp = s(
            v,
            "comp_mode",
            if precision == "P0" { "none" } else { "lanes" },
        );
        if comp == "offload" && lanes.len() < 2 {
            return Err("offload requires two cores and a distinct helper".into());
        }
        let bits = if precision == "P0" {
            16
        } else {
            n(v, "main_bits", 4)
        };
        let factor_a = s(v, "factor_a", "mxint4");
        let factor_b = s(v, "factor_b", "bf16");
        if precision == "P2" && factor_a == "bf16" && comp != "none" {
            return Err("P2 A must be MXINT4 or MXINT8-split".into());
        }
        if precision == "P2" && comp == "kext" {
            return Err("kext is only legal in P1".into());
        }
        if precision == "P2" && (comp == "separate" || comp == "offload") && factor_b != "mxint4" {
            return Err("P2 separate/offload requires B MXINT4".into());
        }
        if !["none", "lanes", "separate", "kext", "offload"].contains(&comp.as_str()) {
            return Err("invalid compensation mode".into());
        }
        if precision == "P2" && flag(v, "slack_pack", false) {
            return Err("slack_pack is only legal in P1".into());
        }
        if comp == "lanes" && l == 0 {
            return Err("lanes compensation needs rank_lanes > 0".into());
        }
        let asym = lanes.len() == 2 && lanes[0] != lanes[1];
        let flows = if let Some(a) = v.get("dataflow").and_then(Value::as_array) {
            a.iter()
                .map(|x| x.as_str().unwrap_or("switchable").into())
                .collect()
        } else if asym {
            vec!["ws_group".into(), "is_stream".into()]
        } else {
            vec!["switchable".into(); lanes.len()]
        };
        if flows.len() != lanes.len() {
            return Err("dataflow length differs from core count".into());
        }
        let iso = flag(v, "iso_port", true);
        let g = vector(
            v,
            "wor_tiles",
            if lanes.len() == 1 {
                vec![8]
            } else if asym {
                vec![8, 1]
            } else {
                vec![4, 4]
            },
        );
        let mut rd = vector(
            v,
            "pool_read_per_core",
            if lanes.len() == 1 && iso {
                vec![1024]
            } else {
                vec![512; lanes.len()]
            },
        );
        let mut xbw = vector(
            v,
            "x_port",
            if lanes.len() == 1 && iso {
                vec![640]
            } else if asym {
                vec![512, 128]
            } else {
                vec![320; lanes.len()]
            },
        );
        if !flag(v, "wide_ports", true) {
            for c in 0..lanes.len() {
                rd[c] = if lanes.len() == 1 { 1024 } else { 512 };
                xbw[c] = lanes[c] * 64;
            }
        }
        if [g.len(), rd.len(), xbw.len()]
            .iter()
            .any(|&a| a != lanes.len())
        {
            return Err("per-core resource vector has wrong length".into());
        }
        if rd.iter().chain(xbw.iter()).any(|&x| x == 0) {
            return Err("ports must be positive".into());
        }
        let ranks = |name: &str, default: [usize; 3]| -> Result<[usize; 3], String> {
            match v.get("ranks").and_then(|x| x.get(name)) {
                Some(Value::Array(a)) if a.len() == 3 => Ok([
                    a[0].as_u64().unwrap_or(0) as usize,
                    a[1].as_u64().unwrap_or(0) as usize,
                    a[2].as_u64().unwrap_or(0) as usize,
                ]),
                Some(_) => Err("ranks must be three integers".into()),
                None => Ok(default),
            }
        };
        let scale = if l == 16 { 2 } else { 1 };
        let rt = ranks("routed", [32 * scale, 32 * scale, 24 * scale])?;
        let sh = ranks("shared", [32 * scale, 32 * scale, 48 * scale])?;
        let bw = n(v, "hbm_bytes_per_ns", n(v, "bandwidth", 256));
        let lat = n(v, "hbm_latency_ns", 64) as u64;
        let credits = n(v, "credits", ceil(bw * (lat as usize + 4), 32));
        let pool = n(
            v,
            "pool_bytes",
            if bw > 256 { 128 * 1024 } else { 64 * 1024 },
        );
        let banks = n(v, "pool_banks", 128);
        if bw < 32 || bw % 32 != 0 || credits == 0 || banks == 0 || pool < 4096 {
            return Err("invalid HBM credits, granularity or pool".into());
        }
        let decoder = n(
            v,
            "dec_bw",
            if lanes.len() == 1 && iso { 2048 } else { 1024 },
        );
        let accbw=vector(v,"accumulator_port_bw",lanes.iter().map(|m|(32*m).max(128)).collect());
        if accbw.len()!=lanes.len()||accbw.iter().any(|&v|v==0||v%16!=0){return Err("accumulator_port_bw requires positive per-core 16B-multiple widths".into());}
        let rank_rd=vector(v,"rank_cache_local_read_bw",lanes.iter().map(|m|align(m*l*2,16).max(16)).collect());
        let total_write=(2*rank_rd.iter().sum::<usize>()).min(256);let denom=lanes.iter().sum::<usize>();let words=total_write/16;let floor=lanes.iter().map(|m|words*m/denom).sum::<usize>();
        let rank_wr=vector(v,"rank_cache_local_write_bw",lanes.iter().enumerate().map(|(c,m)|16*(words*m/denom+usize::from(c<words-floor))).collect());
        if rank_rd.len()!=lanes.len()||rank_wr.len()!=lanes.len()||rank_rd.iter().chain(rank_wr.iter()).any(|&v|v==0||v%16!=0){return Err("rank cache R/W ports require per-core positive 16B-multiple widths".into());}
        Ok(Self {
            raw: v.clone(),
            lanes,
            precision,
            bits,
            rank_lanes: l,
            comp,
            factor_a,
            factor_b,
            slack_pack: flag(v, "slack_pack", false),
            z_mode: s(v, "z_mode", "auto"),
            bw,
            lat,
            credits,
            release: s(v, "credit_release", "ingress"),
            pool,
            pool_banks: banks,
            ingress: n(v, "ingress_bytes", 8192),
            flows,
            g,
            rd,
            xbw,
            accbw,
            rank_rd,rank_wr,
            dec: decoder,
            vlen: n(v, "vlen", 64),
            placement: s(v, "placement", s(v, "dispatch", "supply_ipd").as_str()),
            contexts: n(v, "contexts_per_core", 2),
            window: n(v, "window", 8),
            max: n(v, "max_cycles", 200_000_000) as u64,
            trace: flag(v, "record_trace", false),
            trace_limit: n(v, "trace_limit", 20000),
            ideal_hbm: flag(v, "ideal_hbm", false),
            ideal_chip: flag(v, "ideal_onchip", false),
            x_reuse: flag(v, "x_reuse", true),
            w_reuse: flag(v, "w_reuse", true),
            pipeline: flag(v, "pipeline_supply", true),
            inline: flag(v, "inline_silu", true),
            quota: flag(v, "prefetch_quota", true),
            quota_policy:s(v,"quota_policy","little"),
            byte_pool: flag(v, "byte_pool", true),
            control: flag(v, "control_cost", true),
            margin: n(v, "quota_margin", 64),
            guard: flag(v, "tail_guard", true),
            kappa: v.get("guard_kappa").and_then(Value::as_f64).unwrap_or(2.0),
            m0: n(v, "guard_m0", 64) as u64,
            rank_alloc: s(v, "rank_alloc", "static"),
            ranks_rt: rt,
            ranks_sh: sh,
        })
    }
    pub(crate) fn wslot_bytes(&self)->usize {if self.precision=="P2"{align(plan::main_bytes(512,self.bits)+self.rank_lanes*4*2,32)}else{4096}}
    pub(crate) fn wor_slots(&self,c:usize)->usize {
        let total=n(&self.raw,"wor_total_slots",18);let sum=self.g.iter().sum::<usize>().max(1);
        let floor=self.g.iter().map(|g|total*g/sum).sum::<usize>();
        total*self.g[c]/sum+usize::from(c<total-floor)
    }
    pub(crate) fn ranks(&self, shared: bool) -> [usize; 3] {
        if shared { self.ranks_sh } else { self.ranks_rt }
    }
}
#[derive(Default)]
struct CoreStats {
    states: [u64; 9],
    main_issues: u64,
    prepass_issues: u64,
    short_issues: u64,
    useful: u64,
    issued: u64,
    issued_main: u64,
    issued_aux: u64,
    useful_aux: u64,
    u_store_bytes: u64,
    u_fp32_bytes: u64,
    accumulator_peak: usize,
    silu_spill_write_bytes:u64,
    silu_copy_source_read_bytes:u64,
    silu_spill_read_bytes:u64,
    z_write_bytes:u64,
    consumer_source_bytes:u64,
    wor_write_bytes:u64,
    xor_write_bytes:u64,
    xor_read_bytes:u64,
    rank_input_bytes:u64,
    rank_cache_read_bytes:u64,
    rank_cache_write_bytes:u64,
    rank_macs: u64,
    rank_capacity: u64,
    x_bytes: u64,
    wort_read: u64,
    acc_bytes: u64,
    combine_bytes: u64,
    control: u64,
    done: u64,
    switches: u64,
    switch_cycles: u64,
    service_hist: BTreeMap<u64, u64>,
    issue_commit_hist:BTreeMap<u64,u64>,
    pool_wor_hist:BTreeMap<u64,u64>,
}
#[derive(Clone)]
struct Tile {
    spec: plan::TileSpec,
    task: usize,
    core: usize,
    addr: Option<usize>,
    reserved_bytes:usize,
    sent: usize,
    landed: usize,
    read: usize,
    ready: bool,
    loaded: bool,
    released: bool,
    read_start: Option<u64>,
    load_end: u64,
    resident: bool,
    resident_core: usize,
    resident_slots: usize,
    discarded:bool,
}
struct Task {
    expert: usize,
    core: usize,
    plan: Plan,
    resident_tiles:usize,
    offset: usize,
    action: usize,
    tile_in_group: usize,
    row_block: usize,
    pass: usize,
    born: u64,
    z_addr: Option<usize>,
    u_addr: Option<usize>,
    acc_addr:Option<usize>,
    helper_acc:Option<(usize,usize,usize)>,
    helper_copy:Option<(usize,usize,usize,usize)>,
    consumer_read:usize,
    copy_owner_read:usize,
    copy_write:usize,
    copy_ready:Option<u64>,
    spill_written:usize,
    spill_source_read:usize,
    spill_read:usize,
    bound_at: u64,
    predicted_completion: u64,
    start: Option<u64>,
    predicted: u64,
    blocked: u64,
    done: bool,
    admit: usize,
    send: usize,
    bytes_live: usize,
    peak: usize,
    max_tile: usize,
    eligible_since: Option<u64>,
    group_begin: Option<u64>,
    vector_started: bool,
    action_io: usize,
    action_read: usize,
    rank_io:usize,
    context_wait_since:u64,
    group_x_tags: Vec<(usize, usize)>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct XTag {
    task: usize,
    source: usize,
    seg: usize,
    block: usize,
}
struct XSlot {
    tag: Option<XTag>,
    bytes: usize,
    done: usize,
    ready: bool,
    busy_until: u64,
    spans: Vec<(usize, usize)>,
}
struct RankCache {tag:Option<(usize,usize,usize,usize)>,spans:Vec<(usize,usize)>,done:usize,last:u64}
struct Core {
    cur: Option<usize>,
    next: Option<usize>,
    stats: CoreStats,
    slots: Vec<XSlot>,
    quota: usize,
    live: usize,
    landed_stock: usize,
    rho: f64,
    decode_free: u64,
    issue_free: u64,
    acc_free: u64,
    last_flow: String,
    feedback: [f64; 9],
    error: [f64; 9],
    wor_peak: usize,
    wor_live: usize,
    rank_caches:Vec<RankCache>,
    acc_arena:Pool,
    spill_port:BankPort,
    k_order: BTreeMap<(usize, usize, usize, usize), (usize, u64)>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Req {
    tile: usize,
    offset: usize,
    sent: u64,
}
struct Engine {
    cfg: Cfg,
    experts: Vec<Expert>,
    batch: usize,
    hidden: usize,
    topk: usize,
    now: u64,
    pool: Pool,
    z_arena: Pool,
    u_arena: Pool,
    context_switches:u64,
    activation: BankPort,
    combine: BankPort,
    combine_owner: Option<usize>,
    cores: Vec<Core>,
    tasks: Vec<Task>,
    tiles: Vec<Tile>,
    pending: VecDeque<usize>,
    joint_pending:VecDeque<(usize,usize)>,
    joint_ages:Vec<u8>,
    joint_proposals:u64,
    joint_cancellations:u64,
    joint_comparisons:u64,
    input: usize,
    status: Vec<u8>,
    returns: VecDeque<(u64, Req)>,
    arrived: VecDeque<Req>,
    ingress: VecDeque<Req>,
    credit: usize,
    credit_peak: usize,
    ingress_peak: usize,
    hstates: [u64; 7],
    seq: u64,
    rr: usize,
    control_free: u64,
    vector_free: u64,
    binding: Vec<Value>,
    trace: Vec<Value>,
    last_progress: u64,
    lat_ewma: f64,
    requests: u64,
    landed_requests: u64,
    wire_bytes: u64,
    quota_updates: u64,
    pool_quota_denials: u64,
    quota_progress_borrows:u64,
    pool_capacity_denials: u64,
    tail_fallbacks: u64,
    steal_attempts: u64,
    steals: u64,
    steal_waste_bytes:u64,
    pred_error: f64,
    pred_worst: i64,
    completion: Vec<i64>,
    next_bindings: u64,
    helper_free: u64,
    helper_cycles: u64,
    helper_issued: Option<usize>,
    helper_state:Option<(usize,usize)>,
    copy_bytes: u64,
    baseline_unique: u64,
    actual_storage_peak: usize,
    raw_workload: Value,
    numeric: Option<timed_numeric::NumState>,
    functional_error: Option<String>,
    lambda: f64,
    rank_factor_bytes: u64,
    routed_rank_factor_bytes:u64,
    shared_rank_factor_bytes:u64,
}
impl Engine {
    fn new(input: &Value) -> Result<Self, String> {
        let cfg = Cfg::parse(input.get("config").unwrap_or(&Value::Null))?;
        let w = input.get("workload").unwrap_or(input);
        let arr = w
            .get("experts")
            .and_then(Value::as_array)
            .ok_or("workload missing experts")?;
        let mut experts = vec![];
        for e in arr {
            let m = n(e, "Me", 0);
            let h = n(e, "H", n(w, "hidden", 2048));
            let f = n(e, "F", 1408);
            if m == 0 || h == 0 || f == 0 {
                return Err("Me,H,F must be positive".into());
            }
            let tokens: Vec<usize> = e
                .get("token_indices")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_else(|| (0..m).collect());
            if tokens.len() != m {
                return Err("token_indices length must equal Me".into());
            }
            experts.push(Expert {
                id: e
                    .get("id")
                    .and_then(Value::as_i64)
                    .unwrap_or(experts.len() as i64),
                shared: flag(e, "is_shared", false),
                m,
                h,
                f,
                tokens,
            });
        }
        // Preserve the descriptor stream: Shared first, then router expert-ID order.
        experts.sort_by_key(|e| (!e.shared, e.id));
        let batch = n(w, "batch", experts.iter().map(|e| e.m).max().unwrap_or(0));
        let hidden = n(w, "hidden", experts.first().map_or(2048, |e| e.h));
        let topk = n(w, "top_k", 6);
        if experts.iter().any(|e| e.tokens.iter().any(|&t| t >= batch)) {
            return Err("routed token index outside batch".into());
        }
        if batch > 128 {
            return Err("v3 t_chunk >128 is unsupported".into());
        }
        let cores = cfg
            .lanes
            .iter()
            .enumerate().map(|(c,_)| Core {
                cur: None,
                next: None,
                stats: CoreStats::default(),
                slots: (0..2)
                    .map(|_| XSlot {
                        tag: None,
                        bytes: 0,
                        done: 0,
                        ready: false,
                        busy_until: 0,
                        spans: vec![],
                    })
                    .collect(),
                quota: cfg.pool / cfg.lanes.len(),
                live: 0,
                landed_stock: 0,
                rho: 0.0,
                decode_free: 0,
                issue_free: 0,
                acc_free: 0,
                last_flow: String::new(),
                feedback: [1.0; 9],
                error: [0.0; 9],
                rank_caches:(0..2).map(|_|RankCache{tag:None,spans:vec![],done:0,last:0}).collect(),
                wor_peak: 0,
                wor_live: 0,
                acc_arena:Pool::new(1,1),spill_port:BankPort::new(cfg.accbw[c]/16),
                k_order: BTreeMap::new(),
            })
            .collect();
        let count = experts.len();
        let base = experts.iter().map(|e| (6 * e.h * e.f) as u64).sum();
        let mut engine = Self {
            pool: Pool::new(cfg.pool, cfg.pool_banks),
            z_arena: Pool::new(1,1),u_arena: Pool::new(1,1),context_switches:0,
            activation: BankPort::new(64),
            combine: BankPort::new(16),
            combine_owner: None,
            cores,
            tasks: vec![],
            tiles: vec![],
            pending: VecDeque::new(),
            joint_pending:VecDeque::new(),joint_ages:vec![0;count],joint_proposals:0,joint_cancellations:0,joint_comparisons:0,
            input: 0,
            status: vec![0; count],
            returns: VecDeque::new(),
            arrived: VecDeque::new(),
            ingress: VecDeque::new(),
            credit: 0,
            credit_peak: 0,
            ingress_peak: 0,
            hstates: [0; 7],
            seq: 0,
            rr: 0,
            control_free: 0,
            vector_free: 0,
            binding: vec![],
            trace: vec![],
            last_progress: 0,
            lat_ewma: cfg.lat as f64,
            requests: 0,
            landed_requests: 0,
            wire_bytes: 0,
            quota_updates: 0,
            pool_quota_denials: 0,
            quota_progress_borrows:0,
            pool_capacity_denials: 0,
            tail_fallbacks: 0,
            steal_attempts: 0,
            steals: 0,
            steal_waste_bytes:0,
            pred_error: 0.0,
            pred_worst: 0,
            completion: vec![],
            next_bindings: 0,
            helper_free: 0,
            helper_cycles: 0,
            helper_issued: None,helper_state:None,
            copy_bytes: 0,
            baseline_unique: base,
            raw_workload: w.clone(),
            numeric: timed_numeric::NumState::new(
                input.get("numeric_payload").unwrap_or(&Value::Null),
            )?,
            functional_error: None,
            lambda: input
                .get("config")
                .and_then(|v| v.get("lambda_before").or_else(||v.get("lambda0")))
                .and_then(Value::as_f64)
                .unwrap_or(1.0),
            rank_factor_bytes: 0,
            routed_rank_factor_bytes:0,shared_rank_factor_bytes:0,
            actual_storage_peak: 0,
            cfg,
            experts,
            batch,
            hidden,
            topk,
            now: 0,
        };
        engine.check_storage()?;
        if engine.cfg.g.iter().any(|&g|g==0||g>8)||n(&engine.cfg.raw,"wor_total_slots",18)!=18{return Err("v3 fixed WOR provision is 18 total slots and group sizes in1..=8".into());}
        let lambda0=engine.cfg.raw.get("lambda0").and_then(Value::as_f64).unwrap_or(1.0);if !engine.lambda.is_finite()||engine.lambda<lambda0/100.||engine.lambda>lambda0*100.{engine.lambda=lambda0;}
        let storage=engine.storage();engine.z_arena=Pool::new(storage["structures"]["Z_dense"].as_u64().unwrap()as usize+storage["structures"]["Z_stream"].as_u64().unwrap()as usize,1);engine.u_arena=Pool::new(storage["structures"]["U_bf16"].as_u64().unwrap()as usize+storage["structures"]["U_d_fp32"].as_u64().unwrap()as usize,1);
        for c in 0..engine.cores.len(){engine.cores[c].acc_arena=Pool::new(engine.core_acc_capacity(c),1);}
        Ok(engine)
    }
    fn storage(&self) -> Value {
        let t = n(
            &self.cfg.raw,
            "t_chunk",
            if self.cfg.precision == "P2" && self.cfg.rank_lanes <= 8 && self.cfg.bw <= 256 {
                128
            } else {
                96
            },
        );
        let l = self.cfg.rank_lanes;
        let full = self.cfg.z_mode == "full" || (self.cfg.z_mode == "auto" && t <= 64);
        let sh = self.experts.iter().find(|e| e.shared).map_or(
            self.experts.iter().map(|e| e.f).max().unwrap_or(1408),
            |e| e.f,
        );
        let rt = self
            .experts
            .iter()
            .filter(|e| !e.shared)
            .map(|e| e.f)
            .max()
            .unwrap_or(sh);
        let sm = 4;
        let ranksh = self.cfg.ranks_sh;
        let rankrt = self.cfg.ranks_rt;
        let bf = self.cfg.precision != "P2";
        let tile = if bf {
            4096
        } else {
            align(plan::main_bytes(512, self.cfg.bits) + l * 4 * 2, 32)
        };
        let specialized = self.cfg.flows.iter().any(|f| f == "is_stream");
        let wor = n(&self.cfg.raw,"wor_total_slots",18)*tile;
        let xreg = if specialized {
            2 * self.cfg.lanes[0] * 1024 + 2 * sm * 1024
        } else {
            self.cfg.lanes.iter().map(|m| 2 * m * 1024).sum()
        };
        let acc = if specialized {
            2 * t * 32 * 4 + align(sm * (2 * rt).max(self.hidden) * 4, 4096)
        } else {
            self.cfg
                .lanes
                .iter()
                .map(|&m| {
                    (2 * t * 32 * 4).max(align(m * (2 * rt).max(self.hidden) * 4, 4096))
                })
                .sum()
        };
        let ub = t * ranksh.iter().sum::<usize>() * 2 + sm * rankrt.iter().sum::<usize>() * 2;
        let up = t * ranksh[2] * 4 + sm * rankrt[2] * 4;
        let x = t * self.hidden * 2;
        let z = if full { t * sh * 2 } else { 2 * t * 512 * 2 };
        let zs = sm * rt * 2;
        let cmb = t * self.hidden * 4;
        let route = t * self.topk * 16 + 64 * 64;
        let control = 16 * 1024;
        let structures = json!({"ingress_fifo":self.cfg.ingress,"landing_pool":self.cfg.pool,"WOR":wor,"XOR":xreg,"accumulator":acc,"U_bf16":ub,"U_d_fp32":up,"X_activation":x,"Z_dense":z,"Z_stream":zs,"combine_fp32":cmb,"route_state":route,"control":control});
        let total = self.cfg.ingress
            + self.cfg.pool
            + wor
            + xreg
            + acc
            + ub
            + up
            + x
            + z
            + zs
            + cmb
            + route
            + control;
        json!({"structures":structures,"total_bytes":total,"capacity_bytes":2158592,"slack_bytes":2158592i64-total as i64,"fits":total<=2158592,"t_chunk_capacity":t,"actual_tokens":self.batch,"pool_write_bytes_per_cycle":n(&self.cfg.raw,"pool_write_bw",self.cfg.bw),"z_mode":if full{"full"}else{"streamed"},"dot_latency_cycles":n(&self.cfg.raw,"dot_latency",20),"installed_wor_slots_per_core":(0..self.cores.len()).map(|c|self.cfg.wor_slots(c)).collect::<Vec<_>>(),"wor_slot_bytes":self.cfg.wslot_bytes(),"rank_operand_cache_reserved_in_control_bytes":self.rank_cache_budget(),"rank_operand_cache_entries_per_core":2,"rank_cache_physical_type":"banked SRAM in fixed 16KiB control reserve; projection-major row-striped 16B banks, 1R1W; local read stage included in dot pipeline","rank_cache_local_read_bytes_per_cycle":(0..self.cores.len()).map(|c|self.rank_cache_read_bw(c)).collect::<Vec<_>>(),"rank_cache_local_write_bytes_per_cycle":(0..self.cores.len()).map(|c|self.rank_cache_write_bw(c)).collect::<Vec<_>>(),"rank_cache_bank_counts":(0..self.cores.len()).map(|c|ceil(self.rank_cache_write_bw(c),16)).collect::<Vec<_>>(),"context_switch_policy":s(&self.cfg.raw,"context_switch_policy","starved"),"quota_policy":self.cfg.quota_policy,"prefetch_quota_enabled":self.cfg.quota,"pipeline_supply_enabled":self.cfg.pipeline,"rank_lut_bytes":if self.cfg.rank_alloc=="gate_weighted"{6186}else{0},"rank_lut_energy_format":"BF16","rank_lut_byte_cost_format":"uint32","control_used_bytes":self.control_usage(),"silu_mode":if self.cfg.inline{"inline RF to SiLU"}else{"deferred group-local WS spill or existing IS FP32 backing"},"silu_backing_read_write_port_per_core_bytes_per_cycle":self.cfg.accbw,"silu_backing_bank_count":self.cfg.accbw.iter().map(|w|w/16).collect::<Vec<_>>(),"accumulator_port_bytes_per_cycle":self.cfg.accbw,"accumulator_capacity_bytes_per_core":(0..self.cores.len()).map(|c|self.core_acc_capacity(c)).collect::<Vec<_>>(),"accumulator_source_and_rmw_share_port":true,"accumulator_source_port_contract":"fixed max(128,32*M)B/cycle shared private RMW/source port; source reads wait all prior dot/acc commits, each core executes one actor at a time","U_bf16_layout":"rank lanes grouped by K-segment interleaving, tails contiguous; U_d FP32 partials retain logical rank order","combine_lease_register_bytes":8,"WOR_rankB_reserved_in_control":if self.cfg.precision=="P1"{18*l*4*2}else{0},"main_multipliers":self.cfg.lanes.iter().sum::<usize>()*4*512,"rank_multipliers":self.cfg.lanes.iter().sum::<usize>()*4*l,"rank_multipliers_installed":self.cfg.lanes.iter().sum::<usize>()*4*l,"rank_multipliers_active":if self.cfg.comp=="lanes"{self.cfg.lanes.iter().sum::<usize>()*4*l}else{0},"port_pool_read_bytes_per_cycle":self.cfg.rd.iter().sum::<usize>(),"port_x_bytes_per_cycle":self.cfg.xbw.iter().sum::<usize>(),"pool_banks":self.cfg.pool_banks,"bank_word_bytes":16})
    }
    fn check_storage(&self) -> Result<(), String> {
        let b = self.storage();
        if self.control_usage()>16384{return Err(format!("control reserve exceeded: {} >16384",self.control_usage()));}
        if self.batch > b["t_chunk_capacity"].as_u64().unwrap_or(0) as usize {
            return Err("actual tokens exceed frozen t_chunk capacity".into());
        }
        if b["fits"] != true {
            return Err(format!("v3 physical storage budget exceeded: {}", b));
        }
        if self.cfg.rank_alloc == "gate_weighted"
            && self.cfg.raw.get("rank_energy_table").is_none()
            && self.raw_workload.get("rank_energy_table").is_none()
        {
            return Err("gate_weighted requires calibrated rank_energy_table; missing calibration is not simulated".into());
        }
        if self.cfg.rank_alloc == "gate_weighted" {
            let table = self
                .cfg
                .raw
                .get("rank_energy_table")
                .or_else(|| self.raw_workload.get("rank_energy_table"))
                .unwrap();
            for e in self.experts.iter().filter(|e| !e.shared) {
                for name in ["gate", "up", "down"] {
                    let vals = table
                        .get(e.id.to_string())
                        .and_then(|d| d.get(name))
                        .or_else(|| {
                            table
                                .get("tail_energy_per_projection")
                                .and_then(|d| d.get(name))
                                .and_then(|a| a.get(e.id.max(0) as usize))
                        });
                    if vals
                        .and_then(Value::as_array)
                        .is_none_or(|a| a.len() != 5 || a.iter().any(|v| v.as_f64().is_none()))
                    {
                        return Err(format!(
                            "calibration missing five candidate energies for expert {} projection {}",
                            e.id, name
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    fn log(&mut self, v: Value) {
        if self.cfg.trace && self.trace.len() < self.cfg.trace_limit {
            self.trace.push(v);
        }
    }
    fn core_acc_capacity(&self,c:usize)->usize {
        let t=n(&self.cfg.raw,"t_chunk",if self.cfg.precision=="P2"&&self.cfg.rank_lanes<=8&&self.cfg.bw<=256{128}else{96});
        let rt=self.experts.iter().filter(|e|!e.shared).map(|e|e.f).max().unwrap_or_else(||self.experts.iter().map(|e|e.f).max().unwrap_or(1408));
        if self.cfg.flows.iter().any(|f|f=="is_stream") {if self.cfg.flows[c]=="is_stream"{align(4*(2*rt).max(self.hidden)*4,4096)}else{2*t*32*4}}
        else{(2*t*32*4).max(align(self.cfg.lanes[c]*(2*rt).max(self.hidden)*4,4096))}
    }
    fn flow(&self, e: usize, c: usize) -> String {
        let f = &self.cfg.flows[c];
        if f == "switchable" {
            if self.experts[e].m <= self.cfg.lanes[c] && self.experts[e].m*(2*self.experts[e].f).max(self.experts[e].h)*4<=self.core_acc_capacity(c) {
                "is_stream".into()
            } else {
                "ws_group".into()
            }
        } else {
            f.clone()
        }
    }
    fn legal(&self, e: usize, c: usize) -> bool {
        let ex = &self.experts[e];
        if self.cfg.flows[c] == "is_stream" && ex.m > 4 {
            return false;
        }
        if ex.shared && self.cores.len() > 1 {
            let big = self
                .cfg
                .lanes
                .iter()
                .enumerate()
                .max_by_key(|&(i, m)| (*m, std::cmp::Reverse(i)))
                .unwrap()
                .0;
            return c == big;
        }
        true
    }
    fn estimate(&self, e: usize, c: usize) -> u64 {
        let x = &self.experts[e];
        let nb = ceil(x.m, self.cfg.lanes[c]);
        let ranks = if self.cfg.comp == "none" || self.cfg.precision == "P0" {
            [0; 3]
        } else {
            self.cfg.ranks(x.shared)
        };
        let bytes = if self.cfg.precision == "P0" {
            6 * x.h * x.f
        } else {
            (6 * x.h * x.f * self.cfg.bits) / 16
                + (2 * x.f * ceil(x.h, 32) + x.h * ceil(x.f, 32))
                + 2 * (ranks[0] + ranks[1]) * (x.h + x.f)
                + 2 * ranks[2] * (x.h + x.f)
        };
        let tiles = 2 * ceil(x.f, 4) * ceil(x.h, 512) + ceil(x.h, 4) * ceil(x.f, 512);
        let feed = ceil(bytes, self.cfg.rd[c]);
        let compute = tiles * nb;
        let xfeed = if self.flow(e, c) == "is_stream" {
            ceil(x.m * (x.h + x.f) * 2, self.cfg.xbw[c])
        } else {
            ceil(
                (2 * ceil(x.f, 4) + ceil(x.h, 4)) * nb * self.cfg.lanes[c] * 1024,
                self.cfg.g[c] * self.cfg.xbw[c],
            )
        };
        let bin = (usize::BITS - x.m.max(1).leading_zeros()) as usize - 1;
        let on = compute.max(feed).max(xfeed) as f64 * self.cores[c].feedback[bin.min(8)];
        let effective = self
            .cfg
            .bw
            .min(self.cfg.credits * 32 / self.cfg.lat.max(1) as usize)
            .max(1);
        ((on.max(bytes as f64 / effective as f64)).ceil() as u64) + self.cfg.lat + 16
    }
    fn allocate_ranks(&self, e: usize) -> [usize; 3] {
        let expert = &self.experts[e];
        let table = self
            .cfg
            .raw
            .get("rank_energy_table")
            .or_else(|| self.raw_workload.get("rank_energy_table"))
            .unwrap();
        let data = &table[expert.id.to_string()];
        let gates = self.raw_workload["experts"]
            .as_array()
            .and_then(|a| a.iter().find(|v| v["id"].as_i64() == Some(expert.id)))
            .and_then(|v| v.get("gate_weights").or_else(||v.get("route_scores")))
            .and_then(Value::as_array);
        let weight = gates.map_or(expert.m as f64, |a| {
            a.iter().filter_map(Value::as_f64).map(|g| g * g).sum()
        });
        let mut out = [0; 3];let mut common_scores=[0.;5];let mut common_ranks=[[0usize;3];5];
        for (p, name) in ["gate", "up", "down"].iter().enumerate() {
            let k = if p == 2 { expert.f } else { expert.h };
            let ncol = if p == 2 { expert.h } else { expert.f };
            let cap = self.cfg.rank_lanes * ceil(k, 512);
            let values = data
                .get(*name)
                .or_else(|| {
                    table
                        .get("tail_energy_per_projection")
                        .and_then(|v| v.get(*name))
                        .and_then(|v| v.get(expert.id.max(0) as usize))
                })
                .and_then(Value::as_array);
            let mut best = f64::INFINITY;
            for (i, r) in [0usize, 8, 16, 24, 32].iter().enumerate() {
                let calibrated_cap = table
                    .get("capacity_per_projection")
                    .and_then(|v| v.get(*name))
                    .and_then(|v| v.get(expert.id.max(0) as usize))
                    .and_then(Value::as_u64)
                    .map_or(cap, |n| n as usize);
                let r=(*r).min(cap.min(calibrated_cap));common_ranks[i][p]=r;
                if r%8!=0{common_scores[i]=f64::INFINITY;continue;}
                let lookup=r/8;
                let Some(energy) = values.and_then(|v| v.get(lookup)).and_then(Value::as_f64) else {common_scores[i]=f64::INFINITY;continue;};
                let bytes = table
                    .get("factor_bytes_per_projection")
                    .and_then(|v| v.get(*name))
                    .and_then(|v| v.get(expert.id.max(0) as usize))
                    .and_then(|v| v.get(lookup))
                    .and_then(|v|v.as_u64().map(|n|n as usize).or_else(||v.as_f64().filter(|n|n.is_finite()&&*n>=0.&&n.fract()==0.).map(|n|n as usize)))
                    .unwrap_or(2*r*(k+ncol));
                let bits=(energy as f32).to_bits();let energy=f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000)as f64;
                let objective = weight * energy + self.lambda * bytes as f64;common_scores[i]+=objective;
                if objective < best {
                    best = objective;
                    out[p] = r;
                }
            }
        }
        if s(&self.cfg.raw,"rank_selection","common")!="independent_projection"{let best=(0..5).min_by(|&a,&b|common_scores[a].partial_cmp(&common_scores[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();out=common_ranks[best];}
        out
    }
    fn refill(&mut self) {
        while self.pending.len() < self.cfg.window && self.input < self.experts.len() {
            self.pending.push_back(self.input);
            self.input += 1;
        }
    }
    fn bind(&mut self) {
        self.refill();
        if self.now < self.control_free || self.pending.is_empty() {
            return;
        }
        let mut candidates = vec![];
        for c in 0..self.cores.len() {
            let core = &self.cores[c];
            if self.helper_owner().is_some_and(|o| o != c) && core.cur.is_none() {
                continue;
            }
            let isnext = core.cur.is_some();
            if isnext && (self.cfg.contexts < 2 || core.next.is_some()) {
                continue;
            }
            if isnext {
                let current = &self.tasks[core.cur.unwrap()];
                let remain = (current.born + current.predicted).saturating_sub(self.now);
                let drain = ceil(core.live, self.cfg.rd[c]) as u64;
                if remain > self.lat_ewma.ceil() as u64 + drain + 5 {
                    continue;
                }
            }
            for (pos, &e) in self.pending.iter().enumerate() {
                if !self.legal(e, c) {
                    continue;
                }
                let pred = self.estimate(e, c);
                let remaining = core.cur.map_or(0, |t| {
                    (self.tasks[t].born + self.tasks[t].predicted).saturating_sub(self.now)
                });
                let score = remaining + pred;
                let affinity = if self.experts[e].m <= self.cfg.lanes[c]
                    && self.cfg.lanes[c] == *self.cfg.lanes.iter().min().unwrap()
                {
                    0
                } else {
                    1
                };
                let candidate = match self.cfg.placement.as_str() {
                    "fifo" => (pos as u64, c as u64, score),
                    "supply_ipd" => {
                        let big = self.cfg.lanes[c] == *self.cfg.lanes.iter().max().unwrap();
                        (
                            if big {
                                128 - self.experts[e].m.min(128)
                            } else {
                                self.experts[e].m
                            } as u64,
                            affinity,
                            score,
                        )
                    }
                    _ => (score, affinity, pos as u64),
                };
                candidates.push((candidate, pos, e, c, pred, isnext));
                if self.cfg.placement == "fifo" {
                    break;
                }
            }
        }
        if self.cfg.placement=="joint" {
            if self.joint_pending.is_empty() {
                let nc=self.cores.len();let view=self.pending.iter().take(8).map(|&e|{
                    let mut potential=0;let mut eligible=0;let mut service=[0;2];let mut finish=[0;2];
                    for c in 0..nc {if self.legal(e,c){potential|=1<<c;service[c]=self.estimate(e,c);finish[c]=self.now+service[c]+self.cores[c].cur.map_or(0,|t|(self.tasks[t].born+self.tasks[t].predicted).saturating_sub(self.now));}
                        if candidates.iter().any(|x|x.2==e&&x.3==c){eligible|=1<<c;}}
                    joint_policy::Candidate{task:e,potential,eligible,service,finish,age:self.joint_ages[e]}
                }).collect::<Vec<_>>();
                let proposal=joint_policy::propose(&view,nc,n(&self.cfg.raw,"joint_age_limit",8).min(255)as u8,flag(&self.cfg.raw,"joint_pairing",true),self.rr);
                if proposal.placements.is_empty(){return;}
                let charged=if self.cfg.control{4+4*view.len()as u64*nc as u64+4*view.len()as u64+proposal.comparisons}else{0};
                self.control_free=self.now+charged;self.joint_proposals+=1;self.joint_comparisons+=proposal.comparisons;
                let charge_core=proposal.placements[0].core;self.cores[charge_core].stats.control+=charged;
                for x in &view {self.joint_ages[x.task]=self.joint_ages[x.task].saturating_add(1);}
                self.joint_pending=proposal.placements.iter().map(|p|(view[p.index].task,p.core)).collect();
                if self.cfg.trace{self.log(json!({"event":"joint_proposal","cycle":self.now,"anchor_expert":proposal.anchor.map(|i|self.experts[view[i].task].id),"age_forced":proposal.age_forced,"placements":self.joint_pending,"eligible_masks":view.iter().map(|x|x.eligible).collect::<Vec<_>>(),"comparisons":proposal.comparisons,"control_cycles":charged}));}
                return;
            }
            let (e,c)=self.joint_pending[0];
            if let Some(chosen)=candidates.iter().cloned().find(|x|x.2==e&&x.3==c){candidates=vec![chosen];self.joint_pending.pop_front();}
            else if self.status[e]!=0||!self.legal(e,c){self.joint_pending.pop_front();self.joint_cancellations+=1;return;}
            else{return;}
        }
        if candidates.is_empty() {
            return;
        }
        if self.cfg.guard
            && self.cfg.placement == "supply_ipd"
            && self.input == self.experts.len()
            && self.pending.len() <= self.cores.len()
        {
            let guarded = candidates
                .iter()
                .cloned()
                .filter(|a| {
                    let c = a.3;
                    let pred = a.4;
                    let other = (0..self.cores.len())
                        .filter(|&o| o != c)
                        .map(|o| {
                            self.cores[o]
                                .cur
                                .map_or(self.now, |t| self.tasks[t].born + self.tasks[t].predicted)
                                + self.cores[o].next.map_or(0, |t| self.tasks[t].predicted)
                        })
                        .max()
                        .unwrap_or(self.now);
                    let bin = ((usize::BITS - self.experts[a.2].m.max(1).leading_zeros()) as usize
                        - 1)
                    .min(8);
                    self.now
                        + (pred as f64 * (1.0 + self.cfg.kappa * self.cores[c].error[bin])).ceil()
                            as u64
                        <= other + self.cfg.m0
                })
                .collect::<Vec<_>>();
            if guarded.is_empty() {
                self.tail_fallbacks += 1;
            } else {
                candidates = guarded;
            }
        }
        candidates.sort_by_key(|a| a.0);
        let (_, pos, e, c, pred, isnext) = candidates[0];
        let flow = self.flow(e, c);
        let g = if flow == "is_stream" {
            1
        } else {
            self.cfg.g[c]
        };
        let mut taskcfg = self.cfg.clone();
        if taskcfg.rank_alloc == "gate_weighted" && !self.experts[e].shared {
            taskcfg.ranks_rt = self.allocate_ranks(e);
        }
        if taskcfg.z_mode == "auto" {
            let zbudget = self.storage()["structures"]["Z_dense"]
                .as_u64()
                .unwrap_or(0) as usize;
            taskcfg.z_mode = if self.experts[e].m * self.experts[e].f * 2 > zbudget {
                "streamed".into()
            } else {
                "full".into()
            };
        }
        let mut plan = match plan::build(&self.experts[e], &taskcfg, c, &flow, g) {
            Ok(x) => x,
            Err(_) => return,
        };
        if flag(&self.cfg.raw, "comp_equal_bytes", false)
            || flag(&self.cfg.raw, "equal_comp_payload", false)
        {
            let mut maximum = plan.bytes;
            for mode in ["none", "lanes", "separate", "kext", "offload"] {
                let mut cmp = taskcfg.clone();
                cmp.comp = mode.into();
                if (cmp.precision == "P2"
                    && (mode == "kext"
                        || ((mode == "separate" || mode == "offload") && cmp.factor_b != "mxint4")))
                    || (mode == "offload" && cmp.lanes.len() < 2)
                {
                    continue;
                }
                if let Ok(variant) = plan::build(&self.experts[e], &cmp, c, &flow, g) {
                    maximum = maximum.max(variant.bytes);
                }
            }
            let padding = maximum - plan.bytes;
            plan.append_padding(padding);
        }
        let waiting = self.cores[c].cur.map_or(0, |t| {
            (self.tasks[t].born + self.tasks[t].predicted).saturating_sub(self.now)
        });
        let predicted_completion = self.now + waiting + pred;
        let taskid = self.tasks.len();
        let offset = self.tiles.len();
        for tile in &plan.tiles {
            self.tiles.push(Tile {
                spec: tile.clone(),
                task: taskid,
                core: c,
                addr: None,reserved_bytes:0,
                sent: 0,
                landed: 0,
                read: 0,
                ready: false,
                loaded: false,
                released: false,
                read_start: None,
                load_end: 0,
                resident: false,
                resident_core: c,
                resident_slots: 0,
                discarded:false,
            });
        }
        let max_tile = plan.tiles.iter().map(|x| x.bytes).max().unwrap_or(4096);
        let cost = if self.cfg.control {7+if self.cfg.rank_alloc=="gate_weighted"&&!self.experts[e].shared{15+ceil(self.experts[e].m,self.cfg.vlen)as u64}else{0}}else{0};
        self.control_free = self.now + cost;
        self.cores[c].stats.control += cost;
        let switching = if self.cores[c].last_flow.is_empty() || self.cores[c].last_flow == flow {
            0
        } else {
            let slot = if self.cfg.precision == "P2" {
                1152
            } else {
                4096
            };
            ceil(2 * self.cfg.g[c] * slot, self.cfg.rd[c]) as u64
                + ceil(2 * self.cfg.lanes[c] * 1024, self.cfg.xbw[c]) as u64
        };
        if switching > 0 {
            self.cores[c].stats.switches += 1;
            self.cores[c].stats.switch_cycles += switching;
        }
        self.cores[c].last_flow = flow.clone();
        self.tasks.push(Task {
            expert: e,
            core: c,
            plan,
            resident_tiles:0,
            offset,
            action: 0,
            tile_in_group: 0,
            row_block: 0,
            pass: 0,
            born: self.now,
            z_addr:None,u_addr:None,acc_addr:None,helper_acc:None,helper_copy:None,consumer_read:0,copy_owner_read:0,copy_write:0,copy_ready:None,spill_written:0,spill_source_read:0,spill_read:0,
            bound_at: self.now,
            predicted_completion,
            start: if isnext { None } else { Some(self.now) },
            predicted: pred,
            blocked: self.now + cost + switching,
            done: false,
            admit: 0,
            send: 0,
            bytes_live: 0,
            peak: 0,
            max_tile,
            eligible_since: None,
            group_begin: None,
            vector_started: false,
            action_io: 0,
            action_read: 0,
            rank_io:0,
            context_wait_since:self.now,
            group_x_tags: vec![],
        });
        let factor_bytes=self.tasks[taskid]
            .plan
            .tiles
            .iter()
            .map(|t| {
                if t.kind == Kind::Main {
                    t.bytes.saturating_sub(align(t.main_bytes, 32)) as u64
                } else if t.kind == Kind::Prepass || t.kind == Kind::Tail {
                    t.bytes as u64
                } else {
                    0
                }
            })
            .sum::<u64>();
        self.rank_factor_bytes+=factor_bytes;
        if self.experts[e].shared{self.shared_rank_factor_bytes+=factor_bytes;}else{self.routed_rank_factor_bytes+=factor_bytes;}
        self.pending.remove(pos);
        self.joint_ages[e]=0;
        self.status[e] = 1;
        if isnext {
            self.cores[c].next = Some(taskid);
            self.next_bindings += 1;
        } else {
            self.cores[c].cur = Some(taskid);
        }
        self.requota();
        let v = json!({"cycle":self.now,"expert_id":self.experts[e].id,"Me":self.experts[e].m,"core":c,"next":isnext,"dataflow":flow,"z_mode":self.tasks[taskid].plan.z_mode,"predicted_cycles":pred,"predicted_completion_cycle":predicted_completion,"legal_core_count":(0..self.cores.len()).filter(|&x|self.legal(e,x)).count(),"bytes":self.tasks[taskid].plan.bytes,"ranks":self.tasks[taskid].plan.ranks,"rank_selection_cycles":if self.cfg.rank_alloc=="gate_weighted"&&!self.experts[e].shared{15+ceil(self.experts[e].m,self.cfg.vlen)}else{0}});
        self.binding.push(v.clone());
        if self.cfg.trace {
            self.log(json!({"event":"bind","data":v}));
        }
        self.last_progress = self.now;
    }
    // Only an unstarted Next can move. Previously accepted prefetches retain
    // their original response leases until drained and are charged as waste;
    // the new owner fetches an independent plan, with at most one steal/layer.
    fn work_steal(&mut self) {
        if self.cores.len() != 2
            || self.steals > 0
            || !flag(&self.cfg.raw, "work_steal", true)
            || self.now < self.control_free
            || self.helper_owner().is_some()
        {
            return;
        }
        for dst in 0..2 {
            let src = 1 - dst;
            if self.cores[dst].cur.is_some() || self.cores[dst].next.is_some() {
                continue;
            }
            let Some(t) = self.cores[src].next else {
                continue;
            };
            self.steal_attempts += 1;
            let task = &self.tasks[t];
            let e = task.expert;
            if task.start.is_some()
                || task.action > 0
                || !self.legal(e, dst)
            {
                continue;
            }
            let pred = self.estimate(e, dst);
            let wait = self.cores[src].cur.map_or(0, |cur| {
                (self.tasks[cur].born + self.tasks[cur].predicted).saturating_sub(self.now)
            });
            let bin =
                ((usize::BITS - self.experts[e].m.max(1).leading_zeros()) as usize - 1).min(8);
            if pred >= wait + task.predicted
                || pred as f64 * (1. + self.cfg.kappa * self.cores[dst].error[bin])
                    > (wait + task.predicted + self.cfg.m0) as f64
            {
                continue;
            }
            let flow = self.flow(e, dst);
            let mut cfg = self.cfg.clone();
            cfg.z_mode = task.plan.z_mode.clone();
            cfg.ranks_rt = task.plan.ranks;
            cfg.ranks_sh = task.plan.ranks;
            let Ok(mut plan) = plan::build(
                &self.experts[e],
                &cfg,
                dst,
                &flow,
                if flow == "is_stream" {
                    1
                } else {
                    self.cfg.g[dst]
                },
            ) else {
                continue;
            };
            plan.append_padding(task.plan.padding_bytes);
            if plan.bytes != task.plan.bytes {
                continue;
            }
            let old_offset=task.offset;let old_count=task.plan.tiles.len();let mut wasted=0;
            for idx in old_offset..old_offset+old_count{
                if self.tiles[idx].addr.is_some(){self.tiles[idx].discarded=true;wasted+=self.tiles[idx].sent as u64;if self.tiles[idx].landed==self.tiles[idx].sent{self.release_discarded(idx);}}
            }
            self.steal_waste_bytes+=wasted;
            let offset=self.tiles.len();
            for spec in &plan.tiles{self.tiles.push(Tile{spec:spec.clone(),task:t,core:dst,addr:None,reserved_bytes:0,sent:0,landed:0,read:0,ready:false,loaded:false,released:false,read_start:None,load_end:0,resident:false,resident_core:dst,resident_slots:0,discarded:false});}
            self.tasks[t].offset=offset;self.tasks[t].admit=0;self.tasks[t].send=0;self.tasks[t].eligible_since=None;
            let cost = if self.cfg.control { 7 } else { 0 };
            self.control_free = self.now + cost;
            self.cores[dst].stats.control += cost;
            self.cores[src].next = None;
            self.cores[dst].cur = Some(t);
            self.tasks[t].core = dst;
            self.tasks[t].plan = plan;
            self.tasks[t].born = self.now;
            self.tasks[t].start = Some(self.now);
            self.tasks[t].predicted = pred;
            let switch =
                if self.cores[dst].last_flow.is_empty() || self.cores[dst].last_flow == flow {
                    0
                } else {
                    ceil(
                        2 * self.cfg.g[dst]
                            * if self.cfg.precision == "P2" {
                                1152
                            } else {
                                4096
                            },
                        self.cfg.rd[dst],
                    ) as u64
                        + ceil(2 * self.cfg.lanes[dst] * 1024, self.cfg.xbw[dst]) as u64
                };
            self.cores[dst].stats.switches += u64::from(switch > 0);
            self.cores[dst].stats.switch_cycles += switch;
            self.cores[dst].last_flow = flow;
            self.tasks[t].blocked = self.now + cost + switch;
            for b in &mut self.binding {
                if b["expert_id"] == self.experts[e].id {
                    b["initial_core"] = json!(src);
                    b["core"] = json!(dst);
                    b["steal_cycle"] = json!(self.now);
                    b["steal_waste_bytes"]=json!(wasted);
                    b["steal_predicted_completion_cycle"] = json!(self.now + pred);
                }
            }
            self.steals += 1;
            self.requota();
            self.last_progress = self.now;
            if self.cfg.trace {
                self.log(json!({"event":"work_steal","cycle":self.now,"expert":self.experts[e].id,"from":src,"to":dst,"accepted_requests":wasted/32,"discarded_hbm_bytes":wasted,"old_response_leases_retained":true,"control_cycles":cost}));
            }
            break;
        }
    }
    fn requota(&mut self) {
        let total = self
            .cores
            .iter()
            .filter(|c| c.cur.is_some() || c.next.is_some())
            .count()
            .max(1);
        let mut rates = vec![0.0; self.cores.len()];
        let mut targets = vec![0.0; self.cores.len()];
        for c in 0..self.cores.len() {
            if let Some(t) = self.cores[c].cur.or(self.cores[c].next) {
                let task = &self.tasks[t];
                let e = &self.experts[task.expert];
                let nb = ceil(e.m, self.cfg.lanes[c]);
                let tile = if self.cfg.precision == "P0" {
                    4096
                } else {
                    1152
                };
                let t_on = (nb as f64).max(tile as f64 / self.cfg.rd[c] as f64).max(
                    if self.cfg.precision == "P1" {
                        4096.0 / self.cfg.dec.max(1) as f64
                    } else {
                        0.0
                    },
                );
                let rho = tile as f64 / t_on.max(tile as f64 / (self.cfg.bw as f64 / total as f64));
                rates[c] = rho;
                targets[c] =
                    rho * (self.lat_ewma + 8.0 + self.cfg.margin as f64) + 2.0 * tile as f64;
            }
        }
        let sum: f64 = targets.iter().sum();
        let available = self.cfg.pool as f64;
        for c in 0..self.cores.len() {
            self.cores[c].rho = rates[c];
            self.cores[c].quota = if !self.cfg.quota {
                if self.cfg.byte_pool {
                    self.cfg.pool
                } else {
                    let (lo,hi)=self.pool_partition(c);hi-lo
                }
            } else {
                (targets[c]
                    * if sum > available {
                        available / sum
                    } else {
                        1.0
                    }) as usize
            };
            self.cores[c].quota = self.cores[c].quota.max(if self.cfg.precision == "P0" {
                4096
            } else {
                2112
            });
        }
        self.quota_updates += 1;
    }
    fn active_tasks(&self) -> Vec<usize> {
        self.cores
            .iter()
            .flat_map(|c| [c.cur, c.next])
            .flatten()
            .collect()
    }
    fn pool_reservation(&self,b:usize)->usize{if self.cfg.byte_pool{b}else{align(b,4096)}}
    fn pool_partition(&self,c:usize)->(usize,usize){let slots=self.cfg.pool/4096;let per=slots/self.cores.len();let lo=c*per*4096;let hi=if c+1==self.cores.len(){slots*4096}else{(c+1)*per*4096};(lo,hi)}
    fn admit(&mut self) {
        let mut active=self.active_tasks();
        active.sort_by_key(|&t|usize::from(self.cores[self.tasks[t].core].cur==Some(t)));
        for tid in active {
            let c = self.tasks[tid].core;
            let offset = self.tasks[tid].offset;
            let no_ahead=!self.cfg.quota||!self.cfg.pipeline;
            if no_ahead && self.cores[c].cur!=Some(tid){continue;}
            if !self.cfg.byte_pool&&self.cores[c].cur!=Some(tid){
                // A static partition cannot lend another core's idle slots.
                // Admit a Next head only as an entire bounded group, and only
                // if both compute contexts can really share the fixed arena.
                // Otherwise an unusable partial Next pins the whole partition.
                let Some(current)=self.cores[c].cur else{continue;};
                if self.accumulator_footprint(current)+self.accumulator_footprint(tid)>self.core_acc_capacity(c){continue;}
                let current_ready=self.group_at(current,self.tasks[current].action).is_none_or(|g|(g.first_tile..g.first_tile+g.tile_count).all(|i|self.tiles[self.tasks[current].offset+i].loaded));
                if !current_ready{continue;}
                let target=self.tasks[tid].plan.actions[self.tasks[tid].action..].iter().find_map(|a|if let Action::Group(g)=a{Some(g.first_tile+g.tile_count)}else{None}).unwrap_or(self.tasks[tid].admit);
                let needed=(self.tasks[tid].admit..target).map(|i|self.pool_reservation(self.tiles[offset+i].spec.bytes)).sum::<usize>();let (lo,hi)=self.pool_partition(c);let free=self.pool.ranges.iter().map(|&(a,b)|(a+b).min(hi).saturating_sub(a.max(lo))).sum::<usize>();
                if needed>free||self.cores[c].live+needed>self.cores[c].quota{continue;}
            }
            let mut count=self.tasks[tid].plan.tiles.len();
            if self.cores[c].cur!=Some(tid){count=self.tasks[tid].plan.actions[self.tasks[tid].action..].iter().find_map(|a|if let Action::Group(g)=a{Some(g.first_tile+g.tile_count)}else{None}).unwrap_or(self.tasks[tid].admit);}
            if no_ahead {
                count=self.group_at(tid,self.tasks[tid].action).map_or(self.tasks[tid].admit,|g|g.first_tile+g.tile_count);
            } else if self.cfg.quota_policy=="fixed_one_tile" {
                // Current can complete its required group; Next may prefetch
                // only its first tile, rather than borrowing the full pool.
                count=if self.cores[c].cur!=Some(tid){1}else{self.group_at(tid,self.tasks[tid].action).map_or(self.tasks[tid].admit,|g|g.first_tile+g.tile_count)};
            }
            while self.tasks[tid].admit < count {
                let idx = offset + self.tasks[tid].admit;
                let b = self.tiles[idx].spec.bytes;
                let reserved=self.pool_reservation(b);
                let limit = if self.cfg.quota {
                    self.cores[c].quota
                } else if self.cfg.byte_pool {
                    self.cfg.pool
                } else {
                    let (lo,hi)=self.pool_partition(c);hi-lo
                };
                // Reserve one legal head tile for the other context. This also
                // works when a whole group exceeds the pool quota. Swapping a
                // full-quota Current for
                // a zero-prefetched Next leaves both contexts unable to progress.
                let other=if self.cores[c].cur==Some(tid){None}else{self.cores[c].cur};
                let current_reserve=other.map_or(0,|o|self.tasks[o].plan.actions[self.tasks[o].action..].iter().find_map(|a|if let Action::Group(g)=a{Some({let i=(g.first_tile+self.tasks[o].tile_in_group).max(self.tasks[o].admit);if i<g.first_tile+g.tile_count{self.pool_reservation(self.tiles[self.tasks[o].offset+i].spec.bytes)}else{0}})}else{None}).unwrap_or(0));
                let required_current=self.cores[c].cur==Some(tid)&&self.group_at(tid,self.tasks[tid].action).is_some_and(|g|idx>=offset+g.first_tile&&idx<offset+g.first_tile+g.tile_count);
                // Quotas are targets for unreserved capacity, not revocable
                // leases. A stage transition may shrink a quota below a
                // bounded Next head already held by this core. The real
                // Current group can borrow actually free pool bytes to break
                // that cycle; future lookahead and Next cannot do so.
                let borrow=self.cfg.quota&&required_current&&self.cores[c].live+reserved>limit;
                if self.cores[c].live + reserved + current_reserve > limit && !borrow {
                    self.pool_quota_denials += 1;
                    break;
                }
                let allocation=if self.cfg.byte_pool{self.pool.alloc(reserved)}else{let (lo,hi)=self.pool_partition(c);self.pool.alloc_partition(reserved,lo,hi)};
                let Some(addr) = allocation else {
                    self.pool_capacity_denials += 1;
                    break;
                };
                if borrow{self.quota_progress_borrows+=1;}
                self.tiles[idx].addr = Some(addr);
                self.tiles[idx].reserved_bytes=reserved;
                self.tasks[tid].admit += 1;
                self.tasks[tid].bytes_live += reserved;
                self.tasks[tid].peak = self.tasks[tid].peak.max(self.tasks[tid].bytes_live);
                self.cores[c].live += reserved;
                self.last_progress = self.now;
                if self.cfg.trace {
                    self.log(json!({"event":"reserve","cycle":self.now,"tile":idx,"addr":addr,"bytes":b,"reserved_bytes":reserved,"owner":c,"private_partition":!self.cfg.byte_pool}));
                }
            }
        }
    }
    fn dma_ready(&self) -> bool {
        let after = n(&self.cfg.raw, "dma_ready_after", 0) as u64;
        let p = n(&self.cfg.raw, "dma_ready_period", 1).max(1) as u64;
        let on = n(&self.cfg.raw, "dma_ready_cycles", 1) as u64;
        self.now >= after && (self.now - after) % p < on
    }
    fn hbm(&mut self) -> usize {
        if !self.dma_ready() {
            self.hstates[1] += 1;
            return 1;
        }
        for t in self.active_tasks() {
            if self.tasks[t].send < self.tasks[t].admit && self.tasks[t].eligible_since.is_none() {
                self.tasks[t].eligible_since = Some(self.now);
            }
        }
        let mut grants = 0;
        let max = if self.cfg.ideal_hbm {
            self.cfg.credits
        } else {
            self.cfg.bw / 32
        };
        for _ in 0..max {
            if self.credit >= self.cfg.credits {
                break;
            }
            let mut best = None;
            for core in &self.cores {
                for tid in [core.cur, core.next].into_iter().flatten() {
                    let task = &self.tasks[tid];
                    if task.send >= task.admit {
                        continue;
                    }
                    let idx = task.offset + task.send;
                    if self.tiles[idx].sent >= self.tiles[idx].spec.bytes {
                        continue;
                    }
                    let c = task.core;
                    let runway = self.cores[c].landed_stock as f64 / self.cores[c].rho.max(1.0);
                    let age = self.now - task.eligible_since.unwrap_or(self.now);
                    let cand = (
                        if age >= 64 { 0 } else { 1 },
                        0,
                        runway as u64,
                        if c == self.rr { 0 } else { 1 },
                        tid,
                        idx,
                        c,
                    );
                    if best.is_none_or(|old| cand < old) {
                        best = Some(cand);
                    }
                }
            }
            let Some((_, _, _, _, tid, idx, c)) = best else {
                break;
            };
            let offset = self.tiles[idx].sent;
            self.tiles[idx].sent += 32;
            self.tasks[tid].eligible_since = Some(self.now);
            self.credit += 1;
            self.credit_peak = self.credit_peak.max(self.credit);
            self.requests += 1;
            self.wire_bytes += 32;
            self.seq += 1;
            let due = self.now + if self.cfg.ideal_hbm { 0 } else { self.cfg.lat };
            self.returns.push_back((
                due,
                Req {
                    tile: idx,
                    offset,
                    sent: self.now,
                },
            ));
            if self.tiles[idx].sent == self.tiles[idx].spec.bytes {
                self.tasks[tid].send += 1;
            }
            self.rr = (c + 1) % self.cores.len();
            grants += 1;
            self.last_progress = self.now;
        }
        let state = if grants > 0 {
            0
        } else if self.credit >= self.cfg.credits {
            if !self.ingress.is_empty() || !self.arrived.is_empty() {
                3
            } else {
                2
            }
        } else if self.pool.used == self.cfg.pool
            || self.active_tasks().iter().any(|&t| {
                self.tasks[t].send == self.tasks[t].admit
                    && self.tasks[t].admit < self.tasks[t].plan.tiles.len()
            })
        {
            4
        } else {
            5
        };
        self.hstates[state] += 1;
        state
    }
    fn returns(&mut self) {
        while self.returns.front().is_some_and(|(t, _)| *t <= self.now) {
            let (_, req) = self.returns.pop_front().unwrap();
            self.arrived.push_back(req);
        }
        while self.ingress.len() * 32 + 32 <= self.cfg.ingress {
            let Some(req) = self.arrived.pop_front() else {
                break;
            };
            if self.cfg.release == "ingress" {
                self.credit -= 1;
            }
            let measured = self.now - req.sent;
            self.lat_ewma = (7.0 * self.lat_ewma + measured as f64) / 8.0;
            self.ingress.push_back(req);
            self.ingress_peak = self.ingress_peak.max(self.ingress.len() * 32);
            self.last_progress = self.now;
        }
    }
    fn release_discarded(&mut self,idx:usize){
        let tile=&mut self.tiles[idx];if tile.released{return;}assert!(tile.discarded&&tile.landed==tile.sent);
        let Some(addr)=tile.addr.take()else{return;};let bytes=tile.spec.bytes;let c=tile.core;let t=tile.task;
        if tile.ready{self.cores[c].landed_stock-=bytes;}
        let reserved=tile.reserved_bytes;self.pool.release(addr,reserved);self.cores[c].live-=reserved;self.tasks[t].bytes_live-=reserved;tile.released=true;
        if self.cfg.trace{self.log(json!({"event":"discard_drain","cycle":self.now,"tile":idx,"accepted_hbm_bytes":self.tiles[idx].sent,"released_reservation_bytes":reserved,"original_response_core":c}));}
    }
    fn land(&mut self, busy: &mut [bool]) {
        let words = if self.cfg.ideal_chip {
            self.cfg.ingress / 32
        } else {
            n(&self.cfg.raw, "pool_write_bw", self.cfg.bw) / 32
        };
        for _ in 0..words {
            let Some(req) = self.ingress.front().cloned() else {
                break;
            };
            let tile = &self.tiles[req.tile];
            let addr = tile.addr.unwrap() + req.offset;
            if !self.cfg.ideal_chip {
                let b0 = (addr / 16) % self.cfg.pool_banks;
                let b1 = ((addr + 16) / 16) % self.cfg.pool_banks;
                if busy[b0] || busy[b1] {
                    self.pool.conflicts += 1;
                    break;
                }
                self.pool.word(addr, busy, true);
                self.pool.word(addr + 16, busy, true);
            } else {
                self.pool.writes += 32;
            }
            self.ingress.pop_front();
            if self.cfg.release == "landing" {
                self.credit -= 1;
            }
            self.tiles[req.tile].landed += 32;
            self.landed_requests += 1;
            if self.tiles[req.tile].discarded {if self.tiles[req.tile].landed==self.tiles[req.tile].sent{self.release_discarded(req.tile);}}
            else if self.tiles[req.tile].landed == self.tiles[req.tile].spec.bytes {
                self.tiles[req.tile].ready = true;
                self.cores[self.tiles[req.tile].core].landed_stock +=
                    self.tiles[req.tile].spec.bytes;
                if self.cfg.trace {
                    self.log(json!({"event":"landed","cycle":self.now,"tile":req.tile}));
                }
            }
            self.last_progress = self.now;
        }
    }
    fn helper_owner(&self) -> Option<usize> {
        if self.cfg.comp != "offload" || self.cores.len() < 2 {
            return None;
        }
        let big = self
            .cfg
            .lanes
            .iter()
            .enumerate()
            .max_by_key(|&(i, m)| (*m, std::cmp::Reverse(i)))
            .unwrap()
            .0;
        let t = self.cores[big].cur?;
        match self.tasks[t].plan.actions.get(self.tasks[t].action) {
            Some(Action::Group(g)) if g.kind != Kind::Main => Some(big),
            _ => None,
        }
    }
    fn service_core(&self, owner: usize, kind: Kind) -> usize {
        if self.cfg.comp == "offload"
            && (kind == Kind::Prepass || kind == Kind::Tail)
            && self.cores.len() > 1
            && owner
                == self
                    .cfg
                    .lanes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(i, m)| (*m, std::cmp::Reverse(i)))
                    .unwrap()
                    .0
        {
            1 - owner
        } else {
            owner
        }
    }
    fn group_at(&self, t: usize, a: usize) -> Option<plan::Group> {
        self.tasks[t].plan.actions.get(a).and_then(|x| match x {
            Action::Group(g) => Some(g.clone()),
            _ => None,
        })
    }
    fn read_pool(&mut self, c: usize, busy: &mut [bool]) {
        let Some(t) = self.cores[c].cur else {
            return;
        };
        if !self.cfg.pipeline&&self.now<self.cores[c].acc_free{return;}
        let action = self.tasks[t].action;
        let first = self.group_at(t, action);
        // Lookahead must not occupy slots needed by an incomplete Current
        // group. A one-buffer IS core otherwise loads a later ready tile
        // while the second tile of its Gate/Up pair is still returning.
        let current_end=first.as_ref().map_or(0,|g|self.tasks[t].offset+g.first_tile+g.tile_count);
        let current_start=first.as_ref().map_or(0,|g|self.tasks[t].offset+g.first_tile);
        let mut group_ids = vec![];
        if let Some(g) = first {
            let start=if self.cfg.pipeline{g.first_tile}else{g.first_tile+self.tasks[t].tile_in_group};
            let end=if self.cfg.pipeline{g.first_tile+g.tile_count}else{start+1};
            group_ids.extend((start..end).map(|x|self.tasks[t].offset+x));
        }
        if self.cfg.pipeline && self.cfg.w_reuse && self.cores[c].next.is_none() {
            let next = (action + 1..self.tasks[t].plan.actions.len())
                .find(|&a| matches!(self.tasks[t].plan.actions[a], Action::Group(_)));
            if let Some(a) = next {
                if let Some(g) = self.group_at(t, a) {
                    group_ids.extend(
                        (g.first_tile..g.first_tile + g.tile_count)
                            .map(|x| self.tasks[t].offset + x),
                    );
                }
            }
        }
        let mut allowance = if self.cfg.ideal_chip {
            usize::MAX / 1024
        } else {
            self.cfg.rd[c]
        };
        for idx in group_ids {
            if allowance < 16 {
                break;
            }
            let tile = &self.tiles[idx];
            if tile.loaded || !tile.ready {
                continue;
            }
            let service = self.service_core(c, tile.spec.kind);
            if service != c && (self.cores[service].cur.is_some()||self.helper_owner()!=Some(c)) {
                continue;
            }
            let slots_needed=if tile.spec.kind==Kind::Padding{0}else if self.cfg.precision=="P2"{ceil(tile.spec.bytes,self.cfg.wslot_bytes())}else{1};
            let current_reserve=if idx>=current_end {(current_start..current_end).filter(|&i|!self.tiles[i].resident&&!self.tiles[i].loaded).map(|i|{let x=&self.tiles[i];if x.spec.kind==Kind::Padding{0}else if self.cfg.precision=="P2"{ceil(x.spec.bytes,self.cfg.wslot_bytes())}else{1}}).sum::<usize>()}else{0};
            if !tile.resident&&self.cores[service].wor_live+slots_needed+current_reserve > self.cfg.wor_slots(service) {
                continue;
            }
            let bytes = tile.spec.bytes;
            let nb = ceil(
                self.experts[self.tasks[t].expert].m,
                self.cfg.lanes[service],
            );
            allowance = allowance.min(self.cfg.rd[service]);
            let repeats = if self.cfg.w_reuse { 1 } else { self.tasks[t].row_block+1 };
            if self.tiles[idx].read_start.is_none() {
                if self.cfg.trace{self.log(json!({"event":"wor_read_start","cycle":self.now,"core":service,"task":t,"tile":idx,"m_block":self.tasks[t].row_block,"previous_acc_commit":self.cores[service].acc_free}));}
                self.tiles[idx].resident=true;self.tasks[t].resident_tiles+=1;self.tiles[idx].resident_core=service;self.tiles[idx].resident_slots=slots_needed;self.cores[service].wor_live+=slots_needed;self.cores[service].wor_peak=self.cores[service].wor_peak.max(self.cores[service].wor_live);
                self.tiles[idx].read_start = Some(self.now);
            }
            while self.tiles[idx].read < bytes * repeats && allowance >= 16 {
                let addr = self.tiles[idx].addr.unwrap() + self.tiles[idx].read % bytes;
                if !self.cfg.ideal_chip && !self.pool.word(addr, busy, false) {
                    break;
                }
                if self.cfg.ideal_chip {
                    self.pool.reads += 16;
                }
                self.tiles[idx].read += 16;
                if self.cfg.precision!="P1"&&self.tiles[idx].spec.kind!=Kind::Padding{self.cores[service].stats.wor_write_bytes+=16;}
                allowance -= 16;
                self.last_progress = self.now;
            }
            if self.tiles[idx].read >= bytes * repeats {
                let decode = if self.cfg.precision == "P1" && !self.cfg.ideal_chip {
                    ceil(
                        if self.tiles[idx].spec.kind == Kind::Main {
                            self.tiles[idx].spec.kv * 4 * 2
                        } else {
                            self.tiles[idx].spec.kv * 4 * 2
                        },
                        self.cfg.dec.max(1),
                    ) as u64
                } else {
                    0
                };
                let end = self.now + 1;
                self.cores[service].decode_free = self.cores[service].decode_free.max(end) + decode;
                self.tiles[idx].load_end = if self.cfg.ideal_chip {
                    self.now
                } else {
                    self.cores[service].decode_free
                };
                self.tiles[idx].loaded = true;
                if self.cfg.precision=="P1"&&self.tiles[idx].spec.kind!=Kind::Padding{self.cores[service].stats.wor_write_bytes+=align(self.tiles[idx].spec.kv*4*2+if self.tiles[idx].spec.kind==Kind::Main{self.tiles[idx].spec.ranks*4*2}else{0},16)as u64;}
                *self.cores[service].stats.pool_wor_hist.entry(self.tiles[idx].load_end-self.tiles[idx].read_start.unwrap()).or_default()+=1;

                let resident = self.cores[service].wor_live;
                self.cores[service].wor_peak = self.cores[service].wor_peak.max(resident);
                assert!(
                    resident <= self.cfg.wor_slots(service),
                    "WOR slot overcommit"
                );
                let final_copy=self.cfg.w_reuse||repeats>=nb;
                if final_copy{let addr=self.tiles[idx].addr.unwrap();let reserved=self.tiles[idx].reserved_bytes;self.pool.release(addr,reserved);self.tiles[idx].released=true;self.cores[c].live-=reserved;self.cores[c].landed_stock-=bytes;self.tasks[t].bytes_live-=reserved;}
                if self.cfg.trace {
                    self.log(json!({"event":"wor_loaded","cycle":self.now,"ready":self.tiles[idx].load_end,"tile":idx,"m_block":self.tasks[t].row_block,"reference_count":if final_copy{0}else{nb-repeats},"pool_released":final_copy,"bytes":bytes,"reserved_bytes":self.tiles[idx].reserved_bytes}));
                }
            }
        }
    }
    fn input_tag(&self, t: usize, g: &plan::Group, b: usize) -> XTag {
        let source = if g.kind == Kind::Tail {
            2 + g.projection
        } else if g.projection == 2 || g.projection == 4 {
            1
        } else {
            0
        };
        XTag {
            task: t,
            source,
            seg: g.kseg,
            block: if self.tasks[t].plan.dataflow == "is_stream" {
                0
            } else {
                b
            },
        }
    }
    fn context_footprint(&self,t:usize)->(usize,usize) {
        let x=&self.tasks[t];let e=&self.experts[x.expert];
        (align(if x.plan.z_mode=="streamed"{e.m*1024*2}else{e.m*e.f*2},16),align(e.m*x.plan.ranks.iter().sum::<usize>()*2+e.m*x.plan.ranks[2]*4,16))
    }
    fn activate_context(&mut self,t:usize)->bool {
        if self.tasks[t].z_addr.is_some(){return true;}
        let c=self.tasks[t].core;let acc=self.accumulator_footprint(t);
        let Some(aa)=self.cores[c].acc_arena.alloc(acc)else{return false;};
        let (z,u)=self.context_footprint(t);
        let Some(za)=self.z_arena.alloc(z)else{self.cores[c].acc_arena.release(aa,acc);return false;};
        let ua=if u>0{match self.u_arena.alloc(u){Some(a)=>a,None=>{self.z_arena.release(za,z);self.cores[c].acc_arena.release(aa,acc);return false;}}}else{0};
        self.tasks[t].z_addr=Some(za);self.tasks[t].u_addr=Some(ua);self.tasks[t].acc_addr=Some(aa);true
    }
    fn z_base(&self,t:usize)->usize {self.batch*self.hidden*2+self.tasks[t].z_addr.expect("executing context has Z reservation")}
    fn u_base(&self,t:usize)->usize {self.batch*self.hidden*2+self.z_arena.capacity+self.tasks[t].u_addr.expect("executing context has U reservation")}
    fn rank_position(&self,t:usize,p:usize,j:usize)->usize {
        if self.cfg.comp!="lanes"||(p==2&&self.tasks[t].plan.z_mode=="streamed"){return j;}
        let e=&self.experts[self.tasks[t].expert];let ns=ceil(if p==2{e.f}else{e.h},512);let fused=self.tasks[t].plan.ranks[p].min(self.cfg.rank_lanes*ns);
        if j>=fused{return j;}
        let seg=j%ns;(0..seg).map(|s|ceil(fused.saturating_sub(s),ns)).sum::<usize>()+j/ns
    }
    fn u_store_spans(&self,t:usize,p:usize,start:usize,cols:usize)->Vec<(usize,usize)> {
        let e=&self.experts[self.tasks[t].expert];let ranks=self.tasks[t].plan.ranks;let nr=ranks.iter().sum::<usize>();let base=self.u_base(t);let mut words=std::collections::BTreeSet::new();
        for row in 0..e.m{for n in start..start+cols{let (proj,j)=if p==4{(2,n)}else if n<ranks[0]{(0,n)}else{(1,n-ranks[0])};let off=ranks[..proj].iter().sum::<usize>()+self.rank_position(t,proj,j);words.insert((base+row*nr*2+off*2)/16*16);}}
        words.into_iter().map(|a|(a,16)).collect()
    }
    fn rank_cache_budget(&self)->usize {
        self.cfg.lanes.iter().enumerate().map(|(c,&m)|{
            let rows=if self.cfg.flows[c]=="is_stream"{4}else{m};
            let gu=2*self.cfg.rank_lanes.max(if self.cfg.slack_pack{self.cfg.ranks_sh[0].max(self.cfg.ranks_sh[1])}else{0});
            let down=self.cfg.rank_lanes.max(if self.cfg.slack_pack{self.cfg.ranks_sh[2]}else{0});
            2*rows*align(gu.max(down)*2,16)
        }).sum()
    }
    fn control_usage(&self)->usize {
        4288+256+2*9*3+128+65+32+320+2*self.cores.len()*8+self.rank_cache_budget()+if self.cfg.precision=="P1"{18*self.cfg.rank_lanes*4*2}else{0}+if self.cfg.rank_alloc=="gate_weighted"{64*3*5*(2+4)+426}else{0}
    }
    fn lambda_next(&self)->f64 {
        if self.cfg.rank_alloc!="gate_weighted"{return self.lambda;}
        let base=self.cfg.raw.get("lambda0").and_then(Value::as_f64).unwrap_or(1.0);
        let eta=self.cfg.raw.get("lambda_eta").and_then(Value::as_f64).unwrap_or(0.5);
        let measured=if s(&self.cfg.raw,"rank_budget_scope","routed")=="routed"{self.routed_rank_factor_bytes}else{self.rank_factor_bytes};
        let budget=n(&self.cfg.raw,"rank_budget_bytes",measured.max(1)as usize).max(1);
        let proposed=self.lambda*(eta*(measured as f64/budget as f64-1.)).exp();
        if !proposed.is_finite()||proposed<base/100.||proposed>base*100.{base}else{proposed}
    }
    fn u_fp32_base(&self,t:usize)->usize {
        let x=&self.tasks[t];let e=&self.experts[x.expert];
        self.u_base(t)+align(e.m*x.plan.ranks.iter().sum::<usize>()*2,16)
    }
    fn accumulator_footprint(&self,t:usize)->usize {
        let e=&self.experts[self.tasks[t].expert];
        if self.tasks[t].plan.dataflow=="is_stream"{e.m*(2*e.f).max(e.h)*4}else{e.m*32*4*if self.cfg.inline{1}else{2}}
    }
    fn z_row_offset(&self,t:usize,row:usize,kseg:usize)->usize {
        let e=&self.experts[self.tasks[t].expert];
        if self.tasks[t].plan.z_mode=="streamed"{row*1024*2+(kseg%2)*512*2}else{row*e.f*2+kseg*512*2}
    }
    fn alternate_context(&mut self,t:usize) {
        if self.cfg.contexts<2||!flag(&self.cfg.raw,"context_interleave",true)||self.cfg.comp=="offload"{return;}
        let c=self.tasks[t].core;
        if self.cores[c].cur!=Some(t){return;}
        let Some(other)=self.cores[c].next else{return;};
        // WOR is a single physical frame, not a free per-context operand
        // snapshot. Finish and retire any prefetched Current group before
        // parking it; otherwise a future half-loaded tile can pin the slot
        // needed by the resumed context's next Gate/Up pair.
        if self.tasks[t].resident_tiles>0{return;}
        let acc_live=self.accumulator_footprint(t)+self.accumulator_footprint(other);
        if acc_live>self.core_acc_capacity(c){return;}
        let nextgroup=self.tasks[other].plan.actions[self.tasks[other].action..].iter().find_map(|a|if let Action::Group(g)=a{Some(g)}else{None});
        // A parked context may own most of the byte quota. Never suspend it
        // after observing only the other context's first prefetched tile:
        // that tile can retire into WOR while the rest of its group cannot
        // reserve a single byte, creating a circular quota wait. A complete
        // ready group and its finite WOR capacity form the resume contract.
        if nextgroup.is_some_and(|g|(g.first_tile..g.first_tile+g.tile_count).any(|i|!self.tiles[self.tasks[other].offset+i].ready&&!self.tiles[self.tasks[other].offset+i].loaded)){return;}
        let needed=nextgroup.map_or(0,|g|(g.first_tile..g.first_tile+g.tile_count).map(|i|{let x=&self.tiles[self.tasks[other].offset+i];if x.resident{0}else if self.cfg.precision=="P2"{ceil(x.spec.bytes,self.cfg.wslot_bytes())}else{1}}).sum::<usize>());
        if self.cores[c].wor_live+needed>self.cfg.wor_slots(c){return;}
        let current_group=self.tasks[t].plan.actions[self.tasks[t].action..].iter().find_map(|a|if let Action::Group(g)=a{Some(g)}else{None});
        let current_ready=current_group.is_none_or(|g|(g.first_tile..g.first_tile+g.tile_count).all(|i|self.tiles[self.tasks[t].offset+i].ready||self.tiles[self.tasks[t].offset+i].loaded));
        let age_default=self.cfg.lat.max(current_group.map_or(1,|g|g.tile_count as u64*ceil(self.experts[self.tasks[t].expert].m,self.cfg.lanes[c])as u64));
        let age=n(&self.cfg.raw,"context_aging",age_default as usize)as u64;let aged=self.now.saturating_sub(self.tasks[other].context_wait_since)>=age;
        if s(&self.cfg.raw,"context_switch_policy","starved")!="alternating"&&current_ready&&!aged{return;}
        if !self.activate_context(other){return;}
        self.tasks[t].context_wait_since=self.now;
        self.cores[c].cur=Some(other);self.cores[c].next=Some(t);
        if self.tasks[other].start.is_none(){self.tasks[other].start=Some(self.now);self.tasks[other].born=self.now;}
        let mut cost=if self.cfg.control{1}else{0};
        if self.tasks[t].plan.dataflow!=self.tasks[other].plan.dataflow {let slot=self.cfg.wslot_bytes();cost+=ceil(2*self.cfg.g[c]*slot,self.cfg.rd[c])as u64+ceil(2*self.cfg.lanes[c]*1024,self.cfg.xbw[c])as u64;self.cores[c].stats.switches+=1;self.cores[c].stats.switch_cycles+=cost;}
        self.tasks[other].blocked=self.tasks[other].blocked.max(self.now+cost);self.cores[c].stats.control+=u64::from(self.cfg.control);self.control_free=self.control_free.max(self.now+u64::from(self.cfg.control));self.context_switches+=1;
        if self.cfg.trace{self.log(json!({"event":"context_switch","cycle":self.now,"core":c,"from_task":t,"to_task":other,"cycles":cost,"private_accumulator_bytes":acc_live,"current_full_group_ready":current_ready,"aging_cycles":age,"aged":aged}));}
    }
    fn x_target_tags(&self, c: usize, t: usize, g: &plan::Group) -> [Option<XTag>; 2] {
        let e = &self.experts[self.tasks[t].expert];
        let block = self.tasks[t].row_block;
        let mut out = [None, None];
        for (i, b) in [block, block + 1].into_iter().enumerate() {
            if b < ceil(e.m, self.cfg.lanes[c]) {
                let tag = self.input_tag(t, g, b);
                if i == 0 || out[0] != Some(tag) {
                    out[i] = Some(tag);
                }
            }
        }
        out
    }
    fn x_spans(&self, c: usize, t: usize, g: &plan::Group, tag: XTag) -> Vec<(usize, usize)> {
        let e = &self.experts[self.tasks[t].expert];
        let first = if self.tasks[t].plan.dataflow == "is_stream" {
            0
        } else {
            tag.block * self.cfg.lanes[c]
        };
        let count = if self.tasks[t].plan.dataflow == "is_stream" {
            e.m
        } else {
            (e.m - first).min(self.cfg.lanes[c])
        };
        let kv = g.kv;
        let rank_base = if g.kind == Kind::Tail {
            let k = if g.projection < 2 { e.h } else { e.f };
            let cap = if self.tasks[t].plan.z_mode == "streamed" && g.projection == 2 {
                self.cfg.rank_lanes
            } else {
                self.cfg.rank_lanes * ceil(k, 512)
                    + if self.cfg.slack_pack {
                        ceil(k, 512) * 512 - k
                    } else {
                        0
                    }
            };
            let fused = if self.cfg.comp == "lanes" {
                self.tasks[t].plan.ranks[g.projection].min(cap)
            } else {
                0
            };
            fused
                + (g.kseg - ceil(k, 512))
                    * if self.cfg.comp == "lanes" {
                        self.cfg.rank_lanes.max(1)
                    } else {
                        512
                    }
        } else {
            0
        };
        let urank = self.tasks[t].plan.ranks.iter().sum::<usize>();
        let uoffset = if g.projection == 1 {
            self.tasks[t].plan.ranks[0]
        } else if g.projection == 2 {
            self.tasks[t].plan.ranks[0] + self.tasks[t].plan.ranks[1]
        } else {
            0
        };
        let rank_count = if g.kind == Kind::Main && self.cfg.comp == "kext" {
            self.tiles[self.tasks[t].offset + g.first_tile].spec.ranks
        } else {
            0
        };
        let rank_start = (g.kseg * 512).saturating_sub(if g.projection == 2 { e.f } else { e.h });
        let mut spans = Vec::with_capacity(count * 2);
        for row in first..first + count {
            if kv > 0 {
                let base = match tag.source {
                    0 => e.tokens[row] * self.hidden * 2 + g.kseg * 512 * 2,
                    1 => self.z_base(t) + self.z_row_offset(t,row,g.kseg),
                    _ => self.u_base(t) + row * urank * 2 + (uoffset + rank_base) * 2,
                };
                spans.push((base, align(kv * 2, 16)));
            }
            if rank_count > 0 {
                spans.push((
                    self.u_base(t) + row * urank * 2 + (uoffset + rank_start) * 2,
                    align(rank_count * 2, 16),
                ));
            }
        }
        spans
    }
    fn load_x(&mut self, owner: usize) {
        let Some(t) = self.cores[owner].cur else {
            return;
        };
        let Some(g) = self.group_at(t, self.tasks[t].action) else {
            return;
        };
        if g.kind == Kind::Padding {
            return;
        }
        let c = self.service_core(owner, g.kind);
        if c != owner && self.cores[c].cur.is_some() {
            return;
        }
        if !self.cfg.pipeline&&self.now<self.cores[c].acc_free{return;}
        let mut tags = self.x_target_tags(c, t, &g);
        if !self.cfg.pipeline{tags[1]=None;}
        let mut allowance = self.cfg.xbw[c];
        for tag in tags.into_iter().flatten() {
            let exists = self.cores[c].slots.iter().position(|x| x.tag == Some(tag));
            let slot = exists.or_else(|| {
                self.cores[c].slots.iter().position(|x| {
                    x.busy_until <= self.now && (x.tag.is_none() || !tags.contains(&x.tag))
                })
            });
            let Some(si) = slot else {
                continue;
            };
            if exists.is_none() {
                let spans = self.x_spans(c, t, &g, tag);
                let bytes = spans.iter().map(|s| s.1).sum();
                if c != owner && tag.source < 2 {
                    let unique = (tag.source * 1_000_000 + tag.seg, tag.block);
                    if !self.tasks[t].group_x_tags.contains(&unique) {
                        self.tasks[t].group_x_tags.push(unique);
                        self.copy_bytes += bytes as u64;
                        self.helper_free = self.helper_free.max(self.now) + ceil(bytes, 256) as u64;
                    }
                }
                if self.cfg.trace {
                    self.log(json!({"event":"x_load","cycle":self.now,"expert":self.experts[self.tasks[t].expert].id,"core":c,"source":tag.source,"k_segment":tag.seg,"m_block":tag.block,"spans":spans,"rows":spans.len(),"valid_bytes":bytes,"padding_zero_bytes":self.cfg.lanes[c]*g.kv*2-bytes.min(self.cfg.lanes[c]*g.kv*2)}));
                }
                self.cores[c].slots[si] = XSlot {
                    tag: Some(tag),
                    bytes,
                    done: 0,
                    ready: false,
                    busy_until: 0,
                    spans,
                };
            }
            if self.cores[c].slots[si].ready {
                continue;
            }
            let core = &mut self.cores[c];
            let x = &mut core.slots[si];
            let moved = if self.cfg.ideal_chip {
                let m = x.bytes - x.done;
                x.done = x.bytes;
                m
            } else {
                self.activation
                    .transfer_spans(&x.spans, &mut x.done, allowance, false)
            };
            allowance = allowance.saturating_sub(moved);
            core.stats.x_bytes += moved as u64;core.stats.xor_write_bytes+=moved as u64;
            if x.done >= x.bytes {
                x.ready = true;
            }
            if moved > 0 {
                self.last_progress = self.now;
            }
            if allowance < 16 {
                break;
            }
        }
    }
    fn reset_group(&mut self, t: usize) {
        if let Some(g) = self.group_at(t, self.tasks[t].action) {
            for i in g.first_tile..g.first_tile + g.tile_count {
                let idx = self.tasks[t].offset + i;
                if self.tiles[idx].resident {
                    self.tiles[idx].resident = false;
                    self.tasks[t].resident_tiles-=1;
                    self.cores[self.tiles[idx].resident_core].wor_live -= self.tiles[idx].resident_slots;self.tiles[idx].resident_slots=0;
                }
            }
        }
        self.tasks[t].action += 1;
        self.tasks[t].tile_in_group = 0;
        self.tasks[t].row_block = 0;
        self.tasks[t].pass = 0;
        self.tasks[t].group_begin = None;
        self.tasks[t].vector_started = false;
        self.tasks[t].action_io = 0;
        self.tasks[t].action_read = 0;self.tasks[t].rank_io=0;self.tasks[t].spill_written=0;self.tasks[t].spill_source_read=0;self.tasks[t].spill_read=0;self.tasks[t].consumer_read=0;
    }
    fn rank_cache_read_bw(&self,c:usize)->usize {self.cfg.rank_rd[c]}
    fn rank_cache_write_bw(&self,c:usize)->usize {self.cfg.rank_wr[c]}
    fn load_rank_cache(&mut self,c:usize,t:usize,g:&plan::Group)->bool {
        if g.kind!=Kind::Main||self.cfg.comp!="lanes"{return true;}
        let mask=(g.first_tile..g.first_tile+g.tile_count).filter_map(|i|{let x=&self.tiles[self.tasks[t].offset+i].spec;if x.ranks>0{Some(1usize<<x.projection)}else{None}}).fold(0,|a,b|a|b);
        if mask==0{return true;}
        let is=self.tasks[t].plan.dataflow=="is_stream";let block=if is{0}else{self.tasks[t].row_block};let key=(t,g.kseg,block,mask);
        let index=self.cores[c].rank_caches.iter().position(|v|v.tag==Some(key)).unwrap_or_else(||self.cores[c].rank_caches.iter().enumerate().min_by_key(|(_,v)|(v.tag.is_some(),v.last)).unwrap().0);
        if self.cores[c].rank_caches[index].tag!=Some(key){
            let e=&self.experts[self.tasks[t].expert];let ranks=self.tasks[t].plan.ranks;let nr=ranks.iter().sum::<usize>();let base=self.u_base(t);let start=block*self.cfg.lanes[c];let rows=if is{e.m}else{(e.m-start).min(self.cfg.lanes[c])};let mut words=std::collections::BTreeSet::new();
            for row in start..start+rows{for p in 0..3{if mask&(1<<p)==0{continue;}let segments=ceil(if p==2{e.f}else{e.h},512);let late=p==2&&self.tasks[t].plan.z_mode=="streamed";let fused=ranks[p].min(self.cfg.rank_lanes*if late{1}else{segments});let ids:Vec<_>=if late{(0..fused).collect()}else{(0..fused).filter(|j|j%segments==g.kseg).chain(if self.cfg.slack_pack&&g.kseg+1==segments{fused..ranks[p].min(fused+segments*512-if p==2{e.f}else{e.h})}else{0..0}).collect()};let off=ranks[..p].iter().sum::<usize>();for j in ids{words.insert((base+row*nr*2+(off+self.rank_position(t,p,j))*2)/16*16);}}}
            let spans:Vec<_>=words.into_iter().map(|a|(a,16)).collect();let cache=&mut self.cores[c].rank_caches[index];cache.tag=Some(key);cache.spans=spans;cache.done=0;
        }
        let write_bw=self.rank_cache_write_bw(c);let read_bw=self.rank_cache_read_bw(c);
        let core=&mut self.cores[c];let cache=&mut core.rank_caches[index];let bytes:usize=cache.spans.iter().map(|s|s.1).sum();
        let moved=if self.cfg.ideal_chip{let left=bytes-cache.done;cache.done=bytes;left}else{self.activation.transfer_spans(&cache.spans,&mut cache.done,256.min(write_bw),false)};
        cache.last=self.now;core.stats.rank_input_bytes+=moved as u64;core.stats.rank_cache_write_bytes+=moved as u64;
        if moved>0{self.last_progress=self.now;}
        if cache.done<bytes{return false;}
        let e=&self.experts[self.tasks[t].expert];let start=block*self.cfg.lanes[c];let rows=if is{e.m}else{(e.m-start).min(self.cfg.lanes[c])};let terms=self.tiles[self.tasks[t].offset+g.first_tile+self.tasks[t].tile_in_group].spec.ranks;
        let needed=rows*align(terms*2,16);let moved=(needed-self.tasks[t].rank_io).min(if self.cfg.ideal_chip{needed}else{read_bw});self.tasks[t].rank_io+=moved;core.stats.rank_cache_read_bytes+=moved as u64;self.tasks[t].rank_io==needed
    }
    fn ensure_helper_acc(&mut self,t:usize,c:usize)->bool {
        if self.tasks[t].helper_acc.is_some(){return true;}
        if self.cores[c].cur.is_some(){return false;}
        let e=&self.experts[self.tasks[t].expert];let r=self.tasks[t].plan.ranks;
        let cols=if self.tasks[t].plan.dataflow=="is_stream"{32.max(r[0]+r[1]).max(r[2])}else{32};let size=align(e.m*cols*4,16);
        let Some(addr)=self.cores[c].acc_arena.alloc(size)else{return false;};self.tasks[t].helper_acc=Some((c,addr,size));true
    }
    fn release_consumed_helper_acc(&mut self, t: usize) {
        // The helper's group result is no longer live after its real consumer
        // finishes. Holding this lease until expert completion can deadlock a
        // full-footprint IS expert against the owner's next offloaded group.
        // This changes no transfer: source reads, stores, and delta returns
        // have already completed through their finite physical ports.
        if let Some((c, addr, bytes)) = self.tasks[t].helper_acc.take() {
            self.cores[c].acc_arena.release(addr, bytes);
            if self.cfg.trace {
                self.log(json!({"event":"helper_acc_retire","cycle":self.now,"task":t,"core":c,"bytes":bytes,"consumer_completed":true}));
            }
        }
    }
    fn source_address(&self,t:usize,c:usize)->usize {
        if c==self.tasks[t].core{self.tasks[t].acc_addr.expect("owner RF lease")}else{self.tasks[t].helper_acc.expect("helper RF lease").1}
    }
    fn private_source_read(&mut self,t:usize,c:usize,spans:&[(usize,usize)])->bool {
        if self.now<self.cores[c].acc_free{return false;}
        let bytes=spans.iter().map(|x|x.1).sum::<usize>();
        let moved=if self.cfg.ideal_chip{let n=bytes-self.tasks[t].consumer_read;self.tasks[t].consumer_read=bytes;n}else{self.cores[c].spill_port.transfer_spans(spans,&mut self.tasks[t].consumer_read,self.cfg.accbw[c],false)};
        self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.consumer_source_bytes+=moved as u64;
        if c!=self.tasks[t].core{self.helper_state=Some((c,6));}
        if moved>0{self.last_progress=self.now;}self.tasks[t].consumer_read==bytes
    }
    fn step_helper_copy(&mut self,t:usize,owner:usize)->bool {
        let Some((helper,projection,n,cols))=self.tasks[t].helper_copy else{return true;};
        let e=&self.experts[self.tasks[t].expert];let m=e.m;let f=e.f;let h=e.h;let is=self.tasks[t].plan.dataflow=="is_stream";
        let src=self.source_address(t,helper);let source=(0..m).map(|row|(src+row*32*4+(n%32)*4,align(cols*4,16))).collect::<Vec<_>>();
        if !self.private_source_read(t,helper,&source){return false;}
        let bytes=source.iter().map(|s|s.1).sum::<usize>();
        let due=if let Some(due)=self.tasks[t].copy_ready{due}else{let due=self.now+ceil(bytes,256)as u64;self.tasks[t].copy_ready=Some(due);self.helper_free=self.helper_free.max(due);due};
        if self.now<due||self.now<self.cores[owner].acc_free{return false;}
        let dst=self.source_address(t,owner);let groupcols=(self.cfg.g[owner]/2).max(1)*4;
        let stride=if is{if projection<2{2*f}else{h}}else{32};
        let col=if is{n+if projection==1{f}else{0}}else if projection<2{n%groupcols+projection*groupcols}else{n%(self.cfg.g[owner]*4).min(32)};
        let spans=(0..m).map(|row|(dst+row*stride*4+col*4,align(cols*4,16))).collect::<Vec<_>>();
        if self.tasks[t].copy_owner_read<bytes {
            let moved=if self.cfg.ideal_chip{let left=bytes-self.tasks[t].copy_owner_read;self.tasks[t].copy_owner_read=bytes;left}else{self.cores[owner].spill_port.transfer_spans(&spans,&mut self.tasks[t].copy_owner_read,self.cfg.accbw[owner],false)};
            self.cores[owner].stats.acc_bytes+=moved as u64;if moved>0{self.last_progress=self.now;}if self.tasks[t].copy_owner_read<bytes{return false;}
        }
        let moved=if self.cfg.ideal_chip{let left=bytes-self.tasks[t].copy_write;self.tasks[t].copy_write=bytes;left}else{self.cores[owner].spill_port.transfer_spans(&spans,&mut self.tasks[t].copy_write,self.cfg.accbw[owner],true)};
        self.cores[owner].stats.acc_bytes+=moved as u64;if moved>0{self.last_progress=self.now;}
        if self.tasks[t].copy_write<bytes{return false;}
        if self.cfg.trace{self.log(json!({"event":"helper_delta_return","cycle":self.now,"task":t,"from_core":helper,"to_core":owner,"projection":projection,"n":n,"cols":cols,"bytes":bytes,"source_read_bytes":bytes,"owner_rmw_bytes":bytes*2,"helper_delta_cleared":true}));}
        self.tasks[t].helper_copy=None;self.tasks[t].consumer_read=0;self.tasks[t].copy_owner_read=0;self.tasks[t].copy_write=0;self.tasks[t].copy_ready=None;
        self.release_consumed_helper_acc(t);
        true
    }
    fn step_core(&mut self, c: usize) -> usize {
        let Some(t) = self.cores[c].cur else {
            return 0;
        };
        if !self.activate_context(t){return 7;}
        if !self.step_helper_copy(t,c){return 6;}
        if self.tasks[t].start.is_none() {
            self.tasks[t].start = Some(self.now);
            self.tasks[t].born = self.now;
        }
        if self.now < self.tasks[t].blocked {
            return if self.now < self.control_free { 8 } else { 7 };
        }
        if self.tasks[t].action >= self.tasks[t].plan.actions.len() {
            self.complete(c, t);
            return 0;
        }
        let action = self.tasks[t].plan.actions[self.tasks[t].action].clone();
        let ee=&self.experts[self.tasks[t].expert];
        let e=Expert{id:ee.id,shared:ee.shared,m:ee.m,h:ee.h,f:ee.f,tokens:vec![]};
        match action {
            Action::Drain => {
                let commit = if self.cfg.comp == "offload" && self.cores.len() > 1 {
                    self.cores[c].acc_free.max(self.cores[1 - c].acc_free)
                } else {
                    self.cores[c].acc_free
                };
                if self.now < commit {
                    return 6;
                }
                self.reset_group(t);
                self.tasks[t].blocked = self.now + if self.cfg.ideal_chip { 0 } else { 1 };
                7
            }
            Action::UAccumulate {rank_start,rank_cols,k_segment} => {
                let service=self.service_core(c,Kind::Prepass);
                if self.now<self.cores[service].acc_free{return 6;}
                let rd=self.tasks[t].plan.ranks[2];
                let base=self.u_fp32_base(t);
                let src=self.source_address(t,service);let stride=if self.tasks[t].plan.dataflow=="is_stream"{rd.max(32)}else{32};let col=if self.tasks[t].plan.dataflow=="is_stream"{rank_start}else{rank_start%32};
                let source=(0..e.m).map(|row|(src+row*stride*4+col*4,align(rank_cols*4,16))).collect::<Vec<_>>();
                if !self.private_source_read(t,service,&source){return 6;}
                let spans=(0..e.m).map(|row|(base+row*rd*4+rank_start*4,align(rank_cols*4,16))).collect::<Vec<_>>();
                let bytes:usize=spans.iter().map(|s|s.1).sum();
                if k_segment==0 {self.tasks[t].action_read=bytes;}
                if self.tasks[t].action_read<bytes {
                    let moved=if self.cfg.ideal_chip{let left=bytes-self.tasks[t].action_read;self.tasks[t].action_read=bytes;left}else{self.activation.transfer_spans(&spans,&mut self.tasks[t].action_read,256,false)};
                    self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.u_fp32_bytes+=moved as u64;
                    if moved>0{self.last_progress=self.now;}
                    if self.tasks[t].action_read<bytes{return 6;}
                }
                let moved=if self.cfg.ideal_chip{let left=bytes-self.tasks[t].action_io;self.tasks[t].action_io=bytes;left}else{self.activation.transfer_spans(&spans,&mut self.tasks[t].action_io,256,true)};
                self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.u_fp32_bytes+=moved as u64;
                if moved>0{self.last_progress=self.now;}
                if self.tasks[t].action_io==bytes {
                    if self.cfg.trace{self.log(json!({"event":"u_fp32_accumulate","cycle":self.now,"task":t,"expert":e.id,"rank_start":rank_start,"rank_cols":rank_cols,"k_segment":k_segment,"read_bytes":if k_segment==0{0}else{bytes},"write_bytes":bytes}));}
                    self.release_consumed_helper_acc(t);
                    self.reset_group(t);
                }
                6
            }
            Action::UStore {projection,rank_start,rank_cols} => {
                let service=self.service_core(c,Kind::Prepass);
                if self.now<self.cores[service].acc_free{return 6;}
                let spans=self.u_store_spans(t,projection,rank_start,rank_cols);
                let bytes:usize=spans.iter().map(|s|s.1).sum();
                if projection==3 {
                    let aa=self.source_address(t,service);let is=self.tasks[t].plan.dataflow=="is_stream";let stride=if is{self.tasks[t].plan.ranks[0]+self.tasks[t].plan.ranks[1]}else{32};let col=if is{rank_start}else{rank_start%32};
                    let fp=(0..e.m).map(|row|(aa+row*stride*4+col*4,align(rank_cols*4,16))).collect::<Vec<_>>();let fpbytes=fp.iter().map(|s|s.1).sum::<usize>();
                    if self.tasks[t].action_read<fpbytes {
                        let moved=if self.cfg.ideal_chip{let n=fpbytes-self.tasks[t].action_read;self.tasks[t].action_read=fpbytes;n}else{self.cores[service].spill_port.transfer_spans(&fp,&mut self.tasks[t].action_read,self.cfg.accbw[service],false)};
                        self.cores[service].stats.acc_bytes+=moved as u64;self.cores[service].stats.consumer_source_bytes+=moved as u64;if service!=c&&moved>0{self.helper_state=Some((service,6));}if moved>0{self.last_progress=self.now;}if self.tasks[t].action_read<fpbytes{return 7;}
                        if self.cfg.trace{self.log(json!({"event":"u_prepass_fp32_source_read","cycle":self.now,"task":t,"core":service,"bytes":fpbytes,"rank_start":rank_start,"rank_cols":rank_cols}));}
                    }
                }
                if !self.tasks[t].vector_started {
                    let start=self.now.max(self.vector_free);self.vector_free=start+if self.cfg.ideal_chip{0}else{ceil(e.m*rank_cols,self.cfg.vlen.max(1))as u64};
                    self.tasks[t].vector_started=true;self.tasks[t].blocked=self.vector_free;
                }
                if self.now<self.vector_free{return 7;}
                if projection==4 {
                    let fpbase=self.u_fp32_base(t);let rd=self.tasks[t].plan.ranks[2];
                    let fps=(0..e.m).map(|row|(fpbase+row*rd*4+rank_start*4,align(rank_cols*4,16))).collect::<Vec<_>>();
                    let fpbytes:usize=fps.iter().map(|s|s.1).sum();
                    if self.tasks[t].action_read<fpbytes {
                        let moved=if self.cfg.ideal_chip{let left=fpbytes-self.tasks[t].action_read;self.tasks[t].action_read=fpbytes;left}else{self.activation.transfer_spans(&fps,&mut self.tasks[t].action_read,256,false)};
                        self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.u_fp32_bytes+=moved as u64;
                        if moved>0{self.last_progress=self.now;}
                        if self.tasks[t].action_read<fpbytes{return 7;}
                    }
                }
                let moved=if self.cfg.ideal_chip{let left=bytes-self.tasks[t].action_io;self.tasks[t].action_io=bytes;left}else{self.activation.transfer_spans(&spans,&mut self.tasks[t].action_io,256,true)};
                self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.u_store_bytes+=moved as u64;
                if moved>0{self.last_progress=self.now;}
                if self.tasks[t].action_io==bytes {
                    if self.cfg.trace{self.log(json!({"event":"u_bf16_store","cycle":self.now,"task":t,"expert":e.id,"projection":projection,"rank_start":rank_start,"rank_cols":rank_cols,"bytes":bytes}));}
                    self.release_consumed_helper_acc(t);
                    self.reset_group(t);
                }
                7
            }
            Action::Vector {
                elements,
                z_col,
                z_cols,
            } => {
                {
                    if self.now<self.cores[c].acc_free{return 6;}
                    let is=self.tasks[t].plan.dataflow=="is_stream";
                    let aa=self.tasks[t].acc_addr.expect("finite private accumulator lease");
                    let spans=if is {(0..e.m).flat_map(|row|[(aa+row*e.f*8+z_col*4,align(z_cols*4,16)),(aa+row*e.f*8+(e.f+z_col)*4,align(z_cols*4,16))]).collect::<Vec<_>>()}
                    else{let base=aa+if self.cfg.inline{0}else{e.m*32*4};(0..e.m).map(|row|(base+row*32*4,align(z_cols*8,16))).collect::<Vec<_>>()};
                    let bytes=spans.iter().map(|s|s.1).sum::<usize>();
                    // A deferred WS copy has three real transfers: read
                    // completed RF values, write scratch, then read scratch.
                    if !is&&!self.cfg.inline&&self.tasks[t].spill_source_read<bytes {
                        let source=(0..e.m).map(|row|(aa+row*32*4,align(z_cols*8,16))).collect::<Vec<_>>();
                        let moved=if self.cfg.ideal_chip{let n=bytes-self.tasks[t].spill_source_read;self.tasks[t].spill_source_read=bytes;n}else{self.cores[c].spill_port.transfer_spans(&source,&mut self.tasks[t].spill_source_read,self.cfg.accbw[c],false)};
                        self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.silu_copy_source_read_bytes+=moved as u64;
                        if moved>0{self.last_progress=self.now;}if self.tasks[t].spill_source_read<bytes{return 7;}
                    }
                    // IS producer commits already write the same FP32 SRAM.
                    // WS spills into its charged second group-sized slice.
                    if is||self.cfg.inline{self.tasks[t].spill_written=bytes;}
                    if self.tasks[t].spill_written<bytes {
                        let moved=if self.cfg.ideal_chip{let n=bytes-self.tasks[t].spill_written;self.tasks[t].spill_written=bytes;n}else{self.cores[c].spill_port.transfer_spans(&spans,&mut self.tasks[t].spill_written,self.cfg.accbw[c],true)};
                        self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.silu_spill_write_bytes+=moved as u64;
                        if moved>0{self.last_progress=self.now;}if self.tasks[t].spill_written<bytes{return 7;}
                    }
                    if self.tasks[t].spill_read<bytes {
                        let moved=if self.cfg.ideal_chip{let n=bytes-self.tasks[t].spill_read;self.tasks[t].spill_read=bytes;n}else{self.cores[c].spill_port.transfer_spans(&spans,&mut self.tasks[t].spill_read,self.cfg.accbw[c],false)};
                        self.cores[c].stats.acc_bytes+=moved as u64;self.cores[c].stats.silu_spill_read_bytes+=moved as u64;
                        if moved>0{self.last_progress=self.now;}if self.tasks[t].spill_read<bytes{return 7;}
                        if self.cfg.trace{self.log(json!({"event":"silu_backing_read","cycle":self.now,"task":t,"core":c,"private_accumulator_address":aa,"bytes":bytes,"spilled_write_bytes":if is||self.cfg.inline{0}else{bytes},"existing_is_fp32_backing":is,"inline":self.cfg.inline}));}
                    }
                }
                if !self.tasks[t].vector_started {
                    let words = if self.cfg.ideal_chip {
                        0
                    } else {
                        ceil(elements, self.cfg.vlen.max(1)) as u64
                    };
                    let start = self.now.max(self.vector_free).max(self.cores[c].acc_free);
                    self.vector_free = start + words;
                    self.tasks[t].blocked = self.vector_free;
                    self.tasks[t].vector_started = true;
                    if self.cfg.trace {
                        self.log(json!({"event":"silu_start","cycle":self.now,"expert":e.id,"col":z_col,"cols":z_cols,"elements":elements,"ready":self.vector_free}));
                    }
                    return 7;
                }
                if self.now < self.vector_free {
                    return 7;
                }
                let spans = (0..e.m)
                    .map(|row| {
                        (
                            self.z_base(t) + self.z_row_offset(t,row,z_col/512)+(z_col%512)*2,
                            align(z_cols * 2, 16),
                        )
                    })
                    .collect::<Vec<_>>();
                let bytes:usize=spans.iter().map(|s|s.1).sum();
                let moved = if self.cfg.ideal_chip {
                    bytes - self.tasks[t].action_io
                } else {
                    self.activation
                        .transfer_spans(&spans, &mut self.tasks[t].action_io, 256, true)
                };
                if self.cfg.ideal_chip {
                    self.tasks[t].action_io = bytes;
                }
                if moved > 0 {
                    self.last_progress = self.now;
                }
                self.cores[c].stats.z_write_bytes+=moved as u64;
                if self.tasks[t].action_io >= bytes {
                    if let Some(num) = &mut self.numeric {
                        if let Err(err) = num.on_vector(e.id, z_col, z_cols) {
                            self.functional_error =
                                Some(format!("timed numeric vector violation: {err}"));
                            return 7;
                        }
                    }
                    self.reset_group(t);
                }
                7
            }
            Action::Combine { col, cols,partial,k_segment } => {
                if self.now<self.cores[c].acc_free{return 6;}
                let aa=self.source_address(t,c);let is=self.tasks[t].plan.dataflow=="is_stream";let stride=if is{e.h}else{32};let start=if is{col}else{col%32};
                let source=(0..e.m).map(|row|(aa+row*stride*4+start*4,align(cols*4,16))).collect::<Vec<_>>();
                if !self.private_source_read(t,c,&source){return 6;}
                if self.combine_owner.is_some_and(|owner|owner!=t){return 6;}
                self.combine_owner=Some(t);
                let bytes = align(e.m * cols * 4, 16);
                let tokens=self.experts[self.tasks[t].expert].tokens.clone();
                let spans = tokens
                    .iter()
                    .map(|&token| (token * self.hidden * 4 + col * 4, align(cols * 4, 16)))
                    .collect::<Vec<_>>();
                let action_read = &mut self.tasks[t].action_read;
                let moved = if self.cfg.ideal_chip {
                    bytes.saturating_sub(*action_read)
                } else {
                    self.combine.transfer_spans(&spans, action_read, 128, false)
                };
                if self.cfg.ideal_chip {
                    *action_read = bytes;
                }
                self.cores[c].stats.combine_bytes += moved as u64;
                if *action_read >= bytes {
                    let moved = if self.cfg.ideal_chip {
                        bytes - self.tasks[t].action_io
                    } else {
                        self.combine
                            .transfer_spans(&spans, &mut self.tasks[t].action_io, 128, true)
                    };
                    if self.cfg.ideal_chip {
                        self.tasks[t].action_io = bytes;
                    }
                    self.cores[c].stats.combine_bytes += moved as u64;
                }
                if self.tasks[t].action_io >= bytes {
                    if let Some(num) = &mut self.numeric {
                        if let Err(err) =
                            num.on_combine(e.id, col, cols, &tokens, self.batch, self.hidden,partial)
                        {
                            self.functional_error =
                                Some(format!("timed numeric combine violation: {err}"));
                            return 7;
                        }
                    }
                    if self.cfg.trace{self.log(json!({"event":"combine_commit","cycle":self.now,"task":t,"expert":e.id,"core":c,"col":col,"cols":cols,"partial":partial,"k_segment":k_segment,"leased_atomically":true}));}
                    self.combine_owner=None;
                    self.reset_group(t);
                    self.last_progress = self.now;
                }
                6
            }
            Action::Group(g) => {
                let owner = c;
                let c = self.service_core(owner, g.kind);
                if c != owner && self.cores[c].cur.is_some() {
                    return 7;
                }

                if self.now < self.helper_free && c != owner {
                    return 7;
                }
                if c!=owner&&!self.ensure_helper_acc(t,c){return 7;}
                let idx = self.tasks[t].offset + g.first_tile + self.tasks[t].tile_in_group;
                let tile = &self.tiles[idx];
                if !tile.loaded {
                    if !self.cfg.byte_pool&&self.tasks[t].row_block==0&&self.tasks[t].tile_in_group==0 {self.alternate_context(t);if self.cores[owner].cur!=Some(t){return 7;}}
                    let tile=&self.tiles[idx];
                    return if tile.ready {
                        4
                    } else if tile.sent > 0 {
                        2
                    } else {
                        3
                    };
                }
                if self.now < tile.load_end {
                    return 4;
                }
                let weight_ready=tile.load_end;
                if g.kind == Kind::Padding {
                    self.reset_group(t);
                    self.last_progress = self.now;
                    return 4;
                }

                let tag = self.input_tag(t, &g, self.tasks[t].row_block);
                let Some(si) = self.cores[c]
                    .slots
                    .iter()
                    .position(|s| s.tag == Some(tag) && s.ready)
                else {
                    return 5;
                };
                if self.now < self.cores[c].issue_free || (!self.cfg.pipeline&&self.now<self.cores[c].acc_free) {
                    return 6;
                }
                if self.tasks[t].group_begin.is_none() {
                    self.tasks[t].group_begin = Some(self.now);
                }
                let mblock = self.tasks[t].row_block;
                let rows = (e.m - mblock * self.cfg.lanes[c]).min(self.cfg.lanes[c]);
                assert_eq!(
                    self.tiles[idx].task, t,
                    "tile must retain original owner task"
                );
                let spec = self.tiles[idx].spec.clone();
                if spec.kind != Kind::Padding {
                    let ordinal = if spec.kind == Kind::Prepass {
                        spec.kseg * g.passes + self.tasks[t].pass
                    } else {
                        spec.kseg
                    };
                    let key = (t, spec.projection, spec.n, mblock);
                    let prior = self.cores[c].k_order.get(&key).copied().or_else(|| {
                        if spec.kind == Kind::Tail && c != owner {
                            self.cores[owner]
                                .k_order
                                .get(&(
                                    t,
                                    spec.projection,
                                    spec.n,
                                    mblock * self.cfg.lanes[c] / self.cfg.lanes[owner],
                                ))
                                .copied()
                        } else {
                            None
                        }
                    });
                    if let Some((before, ready)) = prior {
                        if ordinal != before + 1 || ready > self.now {
                            return 6;
                        }
                    } else if ordinal != 0 {
                        return 6;
                    }
                }
                if !self.load_rank_cache(c,t,&g){return if c==owner{5}else{7};}
                if self.numeric.is_some() {
                    if let Some(num) = &mut self.numeric {
                        if let Err(err) = num.on_issue(
                            e.id,
                            &spec,
                            mblock * self.cfg.lanes[c],
                            rows,
                            g.kind,
                            &self.cfg,
                            &self.tasks[t].plan.z_mode,
                        ) {
                            self.functional_error =
                                Some(format!("timed numeric issue violation: {err}"));
                            return 7;
                        }
                    }
                }
                let mut issue_duration = 1;
                if self.tasks[t].plan.dataflow == "legacy" {
                    issue_duration = 31;
                }
                let accbytes = self.cfg.lanes[c] * 4 * 4 * 2;
                let dot_latency = n(&self.cfg.raw, "dot_latency", 20) as u64;
                let acc_cycles=ceil(accbytes,self.cfg.accbw[c])as u64;
                self.cores[c].issue_free = self.now + issue_duration;
                self.cores[c].acc_free =
                    self.cores[c].acc_free.max(self.now + dot_latency) + acc_cycles;
                let commit_elapsed=self.cores[c].acc_free-self.now;*self.cores[c].stats.issue_commit_hist.entry(commit_elapsed).or_default()+=1;
                self.cores[c].slots[si].busy_until = self.now + issue_duration;
                self.cores[c].stats.acc_bytes += accbytes as u64;
                self.cores[c].stats.xor_read_bytes+=(rows*(spec.kv+if self.cfg.comp=="kext"&&g.kind==Kind::Main{spec.ranks}else{0})*2)as u64;
                if spec.kind != Kind::Padding {
                    let commit = self.cores[c].acc_free;
                    self.cores[c].k_order.insert(
                        (t, spec.projection, spec.n, mblock),
                        (
                            if spec.kind == Kind::Prepass {
                                spec.kseg * g.passes + self.tasks[t].pass
                            } else {
                                spec.kseg
                            },
                            commit,
                        ),
                    );
                }
                let real_cols = 4.min(
                    if spec.projection < 2 {
                        e.f
                    } else if spec.projection == 2 {
                        e.h
                    } else {
                        if spec.projection==3{self.tasks[t].plan.ranks[0]+self.tasks[t].plan.ranks[1]}else{self.tasks[t].plan.ranks[2]}
                    }
                    .saturating_sub(spec.n),
                );
                let issued_macs = (self.cfg.lanes[c] * 4 * 512) as u64;
                self.cores[c].stats.issued += issued_macs;
                self.cores[c].stats.wort_read += if self.cfg.precision == "P2" {
                    spec.bytes
                } else {
                    spec.kv * 4 * 2+if spec.kind==Kind::Main{spec.ranks*4*2}else{0}
                } as u64;
                match g.kind {
                    Kind::Main => {
                        self.cores[c].stats.main_issues += 1;
                        self.cores[c].stats.issued_main+=issued_macs;
                        self.cores[c].stats.useful += (rows * real_cols * spec.kv) as u64;
                        self.cores[c].stats.rank_macs += (rows * real_cols * spec.ranks) as u64;
                        self.cores[c].stats.rank_capacity +=
                            (self.cfg.lanes[c] * 4 * self.cfg.rank_lanes) as u64;
                    }
                    Kind::Prepass => {self.cores[c].stats.prepass_issues+=1;self.cores[c].stats.issued_aux+=issued_macs;self.cores[c].stats.useful_aux+=(rows*real_cols*spec.kv)as u64;},
                    Kind::Tail => {self.cores[c].stats.short_issues+=1;self.cores[c].stats.issued_aux+=issued_macs;self.cores[c].stats.useful_aux+=(rows*real_cols*spec.kv)as u64;},
                    Kind::Padding => {}
                };
                if c != owner {
                    self.helper_cycles += 1;
                    self.helper_issued = Some(c);
                }
                if self.cfg.trace {
                    self.log(json!({"event":"issue","cycle":self.now,"expert":e.id,"core":c,"tile":idx,"kind":format!("{:?}",g.kind),"projection":spec.projection,"k_segment":spec.kseg,"n":spec.n,"m_block":mblock,"rows":rows,"rank_terms":spec.ranks,"weight_ready":weight_ready,"x_ready":true,"accumulator_complete":self.cores[c].acc_free,"dot_latency_cycles":dot_latency}));
                }
                self.last_progress = self.now;
                self.tasks[t].rank_io=0;self.tasks[t].pass += 1;
                if self.tasks[t].pass >= g.passes {
                    self.tasks[t].pass = 0;
                    self.tasks[t].tile_in_group += 1;
                }
                if !self.cfg.x_reuse {
                    self.cores[c].slots[si].ready = false;
                    self.cores[c].slots[si].done = 0;
                }
                if self.tasks[t].tile_in_group >= g.tile_count {
                    self.tasks[t].tile_in_group = 0;
                    self.tasks[t].row_block += 1;
                    if !self.cfg.w_reuse&&self.tasks[t].row_block<ceil(e.m,self.cfg.lanes[c]){
                        for i in g.first_tile..g.first_tile+g.tile_count{let idx=self.tasks[t].offset+i;if self.tiles[idx].resident{self.tiles[idx].resident=false;self.tasks[t].resident_tiles-=1;self.cores[self.tiles[idx].resident_core].wor_live-=self.tiles[idx].resident_slots;self.tiles[idx].resident_slots=0;}self.tiles[idx].loaded=false;self.tiles[idx].read_start=None;self.tiles[idx].load_end=0;if self.cfg.trace{self.log(json!({"event":"wor_retire_m_block","cycle":self.now,"tile":idx,"task":t,"m_block":mblock,"next_m_block":self.tasks[t].row_block,"pool_released":self.tiles[idx].released}));}}
                    }
                    if self.tasks[t].row_block >= ceil(e.m, self.cfg.lanes[c]) {
                        let start = self.tasks[t].group_begin.unwrap();
                        *self.cores[c]
                            .stats
                            .service_hist
                            .entry(self.now + 1 - start)
                            .or_default() += 1;
                        let copy = if c != owner && g.kind == Kind::Tail {
                            align(e.m * 4 * 4, 16)
                        } else {
                            0
                        };
                        self.reset_group(t);
                        if copy==0{self.alternate_context(t);}
                        if copy > 0 {
                            self.copy_bytes += copy as u64;
                            self.tasks[t].helper_copy=Some((c,spec.projection,spec.n,4.min(if spec.projection<2{e.f}else{e.h}.saturating_sub(spec.n))));
                            self.tasks[t].blocked=self.cores[c].acc_free;
                        }
                    }
                }
                if c != owner { 7 } else { 1 }
            }
        }
    }
    fn complete(&mut self, c: usize, t: usize) {
        if self.tasks[t].done {
            return;
        }
        let e = self.tasks[t].expert;
        if let Some((helper,aa,size))=self.tasks[t].helper_acc.take(){self.cores[helper].acc_arena.release(aa,size);}
        if let Some(aa)=self.tasks[t].acc_addr.take(){let size=self.accumulator_footprint(t);self.cores[c].acc_arena.release(aa,size);}
        let (z,u)=self.context_footprint(t);if let Some(a)=self.tasks[t].z_addr.take(){self.z_arena.release(a,z);}if u>0{if let Some(a)=self.tasks[t].u_addr.take(){self.u_arena.release(a,u);}}
        self.tasks[t].done = true;
        self.status[e] = 2;
        let elapsed = self.now - self.tasks[t].born;
        let predicted = self.tasks[t].predicted;
        let err = self.now.abs_diff(self.tasks[t].predicted_completion);
        self.pred_error += err as f64;
        self.pred_worst = self
            .pred_worst
            .max(self.now as i64 - self.tasks[t].predicted_completion as i64);
        let bin = ((usize::BITS - self.experts[e].m.max(1).leading_zeros()) as usize - 1).min(8);
        let ratio = elapsed as f64 / predicted.max(1) as f64;
        self.cores[c].feedback[bin] = self.cores[c].feedback[bin] * 0.875 + ratio * 0.125;
        self.cores[c].error[bin] =
            self.cores[c].error[bin] * 0.875 + err as f64 / elapsed.max(1) as f64 * 0.125;
        self.cores[c].stats.done = self.now;
        self.completion.push(self.experts[e].id);
        for binding in &mut self.binding {
            if binding["expert_id"] == self.experts[e].id && binding["core"] == c {
                binding["actual_completion_cycle"] = json!(self.now);
                binding["actual_cycles"] = json!(self.now - self.tasks[t].bound_at);
                binding["actual_execution_cycles"] = json!(elapsed);
                binding["prediction_error_cycles"] =
                    json!(self.now as i64 - self.tasks[t].predicted_completion as i64);
            }
        }
        if self.cfg.trace {
            self.log(json!({"event":"expert_complete","cycle":self.now,"expert":self.experts[e].id,"core":c}));
        }
        self.cores[c].cur = if self.helper_owner().is_some_and(|o| o != c) {
            None
        } else {
            self.cores[c].next.take()
        };
        if let Some(next) = self.cores[c].cur {
            if self.tasks[next].start.is_none(){self.tasks[next].born = self.now;self.tasks[next].start = Some(self.now);}
            self.tasks[next].blocked = self.tasks[next]
                .blocked
                .max(self.now + if self.cfg.control { 1 } else { 0 });
        }
        self.requota();
        self.last_progress = self.now;
    }
    fn run(mut self) -> Result<Value, String> {
        let mut busy = vec![false; self.cfg.pool_banks];
        let physical = self.storage();
        loop {
            if let Some(err) = &self.functional_error {
                return Err(err.clone());
            }
            if self.now > self.cfg.max {
                return Err(format!(
                    "v3 max cycles exceeded: now={} pending={} active={:?} pool={} credits={} trace={:?} tasks={:?}",
                    self.now,
                    self.pending.len(),
                    self.cores.iter().map(|c| c.cur).collect::<Vec<_>>(),
                    self.pool.used,
                    self.credit, self.trace.iter().rev().take(10).collect::<Vec<_>>(), self.tasks.iter().enumerate().filter(|(_,t)|!t.done).map(|(id,t)|(id,self.experts[t.expert].id,t.action,t.tile_in_group,t.row_block,t.pass,self.cores[t.core].slots.iter().map(|s|(s.tag,s.ready,s.done,s.bytes)).collect::<Vec<_>>(),self.cores[t.core].wor_live,format!("{:?}",t.plan.actions.get(t.action)),self.tiles.iter().enumerate().filter(|(_,x)|x.task==id&&x.resident).map(|(i,x)|(i,x.spec.clone())).collect::<Vec<_>>(),self.group_at(id,t.action).map(|g|(g.first_tile..g.first_tile+g.tile_count).map(|i|{let x=&self.tiles[t.offset+i];(i,x.sent,x.landed,x.ready,x.loaded,x.resident,x.released,x.addr)}).collect::<Vec<_>>()))).collect::<Vec<_>>()
                ));
            }
            if self.now.saturating_sub(self.last_progress) > 1_000_000 {
                return Err(format!(
                    "v3 no-progress guard: now={} pool={} credits={} tasks={:?}",
                    self.now,
                    self.pool.used,
                    self.credit,
                    self.tasks
                        .iter()
                        .filter(|t| !t.done)
                        .map(|t| (t.expert, t.action, t.admit, t.send, t.bytes_live))
                        .collect::<Vec<_>>()
                ));
            }
            if self.status.iter().all(|&s| s == 2)
                && self.returns.is_empty()
                && self.arrived.is_empty()
                && self.ingress.is_empty()
                && self.credit == 0
                && self.now >= self.helper_free
            {
                break;
            }
            let active_live=self.z_arena.used+self.u_arena.used;
            let mut reg_active=0;
            for c in 0..self.cores.len(){
                let core=&self.cores[c];
                let acc_live=core.acc_arena.used;
                if acc_live>self.core_acc_capacity(c){return Err(format!("core {c} live accumulator {acc_live} exceeds fixed capacity {}",self.core_acc_capacity(c)));}
                reg_active+=core.wor_live*self.cfg.wslot_bytes()+core.slots.iter().map(|s|s.bytes).sum::<usize>()+acc_live;
                self.cores[c].stats.accumulator_peak=self.cores[c].stats.accumulator_peak.max(acc_live);
            }
            let live = self.batch * self.hidden * 6
                + self.batch * self.topk * 16
                + 4096
                + 16384
                + self.pool.used
                + self.ingress.len() * 32
                + active_live
                + reg_active;
            if live > 2158592 {
                return Err(format!(
                    "actual simultaneous storage exceeds budget: {live}"
                ));
            }
            self.actual_storage_peak = self.actual_storage_peak.max(live);
            busy.fill(false);
            self.activation.clear();
            self.combine.clear();
            for core in &mut self.cores{core.spill_port.clear();}
            self.returns();
            self.land(&mut busy);
            for c in 0..self.cores.len() {
                if self.cores[c].cur.is_none()
                    && self.cores[c].next.is_some()
                    && !self.helper_owner().is_some_and(|o| o != c)
                {
                    self.cores[c].cur = self.cores[c].next.take();
                    if let Some(t) = self.cores[c].cur {
                        if self.tasks[t].start.is_none(){self.tasks[t].born = self.now;self.tasks[t].start = Some(self.now);}
                    }
                }
            }
            self.work_steal();
            self.bind();
            self.admit();
            let h = self.hbm();
            if self.cfg.ideal_hbm {
                self.returns();
                self.land(&mut busy);
            }
            self.helper_issued = None;self.helper_state=None;
            for c in 0..self.cores.len() {
                self.read_pool(c, &mut busy);
                self.load_x(c);
                let state = if let Some((_,state))=self.helper_state.filter(|(core,_)|*core==c){state}else if self.helper_issued == Some(c) {
                    1
                } else {
                    self.step_core(c)
                };
                self.cores[c].stats.states[state] += 1;
                if state == 3 {
                    if self.cfg.trace {
                        self.log(json!({"event":"C3","cycle":self.now,"core":c,"h_state":h}));
                    }
                }
            }
            self.now += 1;
        }
        for (t,task) in self.tasks.iter().enumerate(){assert_eq!(task.resident_tiles,self.tiles.iter().filter(|x|x.task==t&&x.resident).count(),"host resident cache matches actual tile state");}
        if self.z_arena.used!=0||self.u_arena.used!=0||self.cores.iter().any(|c|c.acc_arena.used!=0||c.wor_live!=0){return Err("v3 ends with a live context/operand arena lease".into());}
        if self.pool.used != 0 {
            return Err(format!(
                "v3 ends with {} reserved pool bytes",
                self.pool.used
            ));
        }
        let expected: u64 = self
            .experts
            .iter()
            .map(|e| (3 * e.m * e.h * e.f) as u64)
            .sum();
        let useful = self.cores.iter().map(|c| c.stats.useful).sum::<u64>();
        if useful != expected {
            return Err(format!(
                "v3 useful MAC mismatch: expected {expected}, actual {useful}"
            ));
        }
        let bytes = self.tasks.iter().map(|t| t.plan.bytes).sum::<usize>();
        if bytes as u64+self.steal_waste_bytes != self.wire_bytes {
            return Err(format!(
                "v3 unique byte mismatch plan {bytes} actual {}",
                self.wire_bytes
            ));
        }
        if self.requests != self.landed_requests {
            return Err("v3 request drain mismatch".into());
        }
        if let Some(num)=&self.numeric{num.finish().map_err(|e|format!("timed numeric final invariant: {e}"))?;}
        let effective = self
            .cfg
            .bw
            .min(self.cfg.credits * 32 / self.cfg.lat.max(1) as usize);
        let payload_bytes = bytes
            - self
                .tasks
                .iter()
                .map(|t| t.plan.padding_bytes)
                .sum::<usize>();
        let eta = payload_bytes as f64 / (self.now.max(1) as f64 * effective as f64);
        let hnames = ["H0", "H4", "H1a", "H1b", "H2", "H3", "H_unused"];
        let hmap: BTreeMap<_, _> = hnames
            .iter()
            .zip(self.hstates)
            .map(|(a, b)| (a.to_string(), b))
            .collect();
        let cores=self.cores.iter().enumerate().map(|(i,c)|{let states:BTreeMap<_,_>=c.stats.states.iter().enumerate().map(|(s,n)|(format!("C{s}"),*n)).collect();json!({"m":self.cfg.lanes[i],"dataflow":self.cfg.flows[i],"states":states,"state_sum":c.stats.states.iter().sum::<u64>(),"main_issues":c.stats.main_issues,"prepass_issues":c.stats.prepass_issues,"short_issues":c.stats.short_issues,"useful_macs":c.stats.useful,"issued_macs":c.stats.issued,"issued_main_macs":c.stats.issued_main,"issued_aux_macs":c.stats.issued_aux,"useful_aux_macs":c.stats.useful_aux,"padding_main_macs":c.stats.issued_main-c.stats.useful,"padding_aux_macs":c.stats.issued_aux-c.stats.useful_aux,"u_bf16_store_bytes":c.stats.u_store_bytes,"u_fp32_rmw_bytes":c.stats.u_fp32_bytes,"accumulator_live_peak_bytes":c.stats.accumulator_peak,"silu_spill_write_bytes":c.stats.silu_spill_write_bytes,"silu_copy_source_read_bytes":c.stats.silu_copy_source_read_bytes,"silu_spill_read_bytes":c.stats.silu_spill_read_bytes,"silu_source_read_bytes":c.stats.silu_spill_read_bytes,"Z_write_bytes":c.stats.z_write_bytes,"private_accumulator_address_peak_bytes":c.acc_arena.peak,"consumer_private_source_read_bytes":c.stats.consumer_source_bytes,"WOR_write_bytes":c.stats.wor_write_bytes,"XOR_write_bytes":c.stats.xor_write_bytes,"XOR_array_read_bytes":c.stats.xor_read_bytes,"accumulator_capacity_bytes":self.core_acc_capacity(i),"rank_input_read_bytes":c.stats.rank_input_bytes,"rank_cache_local_read_bytes":c.stats.rank_cache_read_bytes,"rank_cache_local_write_bytes":c.stats.rank_cache_write_bytes,"wor_live_peak_bytes":c.wor_peak*self.cfg.wslot_bytes(),"installed_wor_slots":self.cfg.wor_slots(i),"wor_slot_bytes":self.cfg.wslot_bytes(),"rank_macs":c.stats.rank_macs,"rank_capacity_macs":c.stats.rank_capacity,"x_read_bytes":c.stats.x_bytes,"wor_read_bytes":c.stats.wort_read,"accumulator_rmw_bytes":c.stats.acc_bytes,"accumulator_and_U_transfer_bytes":c.stats.acc_bytes,"combine_rmw_bytes":c.stats.combine_bytes,"control_cycles":c.stats.control,"busy_cycles":c.stats.states[1]+c.stats.states[4]+c.stats.states[5]+c.stats.states[6]+c.stats.states[7],"done_cycle":c.stats.done,"wor_peak_slots":c.wor_peak,"wor_capacity_slots":self.cfg.wor_slots(i),"switches":c.stats.switches,"switch_cycles":c.stats.switch_cycles,"service_histogram":c.stats.service_hist,"group_wall_elapsed_histogram":c.stats.service_hist,"issue_to_accumulator_completion_histogram":c.stats.issue_commit_hist,"pool_to_wor_elapsed_histogram":c.stats.pool_wor_hist,"prediction_feedback":c.feedback,"prediction_error_ewma":c.error})}).collect::<Vec<_>>();
        let legacy_onchip=self.pool.reads+self.cores.iter().map(|c|c.stats.x_bytes+c.stats.wort_read+c.stats.acc_bytes+c.stats.combine_bytes+c.stats.rank_input_bytes+c.stats.z_write_bytes).sum::<u64>()+self.copy_bytes;
        let onchip=legacy_onchip+self.pool.writes+2*self.wire_bytes+self.cores.iter().map(|c|c.stats.wor_write_bytes+c.stats.xor_write_bytes+c.stats.xor_read_bytes+c.stats.rank_cache_read_bytes+c.stats.rank_cache_write_bytes).sum::<u64>();
        let movement=json!({"ingress_FIFO_write_and_read":2*self.wire_bytes,"pool_read":self.pool.reads,"pool_write":self.pool.writes,"activation_X_read":self.cores.iter().map(|c|c.stats.x_bytes).sum::<u64>(),"XOR_write":self.cores.iter().map(|c|c.stats.xor_write_bytes).sum::<u64>(),"XOR_array_read":self.cores.iter().map(|c|c.stats.xor_read_bytes).sum::<u64>(),"WOR_write":self.cores.iter().map(|c|c.stats.wor_write_bytes).sum::<u64>(),"WOR_array_read_and_broadcast":self.cores.iter().map(|c|c.stats.wort_read).sum::<u64>(),"private_accumulator_and_U_transfers":self.cores.iter().map(|c|c.stats.acc_bytes).sum::<u64>(),"combine_read_and_write":self.cores.iter().map(|c|c.stats.combine_bytes).sum::<u64>(),"activation_rank_U_read":self.cores.iter().map(|c|c.stats.rank_input_bytes).sum::<u64>(),"rank_cache_write":self.cores.iter().map(|c|c.stats.rank_cache_write_bytes).sum::<u64>(),"rank_cache_array_read":self.cores.iter().map(|c|c.stats.rank_cache_read_bytes).sum::<u64>(),"activation_Z_write":self.cores.iter().map(|c|c.stats.z_write_bytes).sum::<u64>(),"cross_core_wire_copy":self.copy_bytes});
        let mut report=json!({"schema":"plena_supply_v3_event_v1","arch":"supply_v3","scope":"cycle/event analytical aggregate HBM latency and bandwidth with finite per-32B ingress, physical bank conflict and operand sequencing; not native Ramulator or RTL","numeric_execution":self.numeric.as_ref().map(|n|n.report()),"config":self.cfg.raw,"cycles":self.now,"time_ms_at_1ghz":self.now as f64/1e6,"weight_bytes":bytes as u64+self.steal_waste_bytes,"unique_weight_bytes":bytes-self.tasks.iter().map(|t|t.plan.padding_bytes).sum::<usize>(),"comp_padding_bytes":self.tasks.iter().map(|t|t.plan.padding_bytes).sum::<usize>(),"baseline_bf16_weight_bytes":self.baseline_unique,"weight_retake_bytes":self.steal_waste_bytes,"work_steal_waste_bytes":self.steal_waste_bytes,"useful_macs":useful,"issued_macs":self.cores.iter().map(|c|c.stats.issued).sum::<u64>(),"issued_main_macs":self.cores.iter().map(|c|c.stats.issued_main).sum::<u64>(),"issued_aux_macs":self.cores.iter().map(|c|c.stats.issued_aux).sum::<u64>(),"useful_aux_macs":self.cores.iter().map(|c|c.stats.useful_aux).sum::<u64>(),"padding_main_macs":self.cores.iter().map(|c|c.stats.issued_main-c.stats.useful).sum::<u64>(),"padding_aux_macs":self.cores.iter().map(|c|c.stats.issued_aux-c.stats.useful_aux).sum::<u64>(),"supply_efficiency":eta,"effective_bandwidth_bytes_per_cycle":effective,"achieved_weight_gbps":self.wire_bytes as f64/self.now.max(1)as f64,"dma_transactions_accepted":self.requests,"dma_transactions_landed":self.landed_requests,"credit_peak":self.credit_peak,"ingress_peak_bytes":self.ingress_peak,"pool_peak_bytes":self.pool.peak,"pool_read_bytes":self.pool.reads,"pool_write_bytes":self.pool.writes,"native_expert_bytes":self.tasks.iter().map(|t|t.plan.bytes-t.plan.padding_bytes).sum::<usize>(),"pool_bank_conflicts":self.pool.conflicts,"activation_bank_conflicts":self.activation.conflicts,"combine_bank_conflicts":self.combine.conflicts,"onchip_traffic_bytes":onchip,"onchip_bytes_per_useful_mac":onchip as f64/useful.max(1)as f64,"hbm_bytes_per_token":self.wire_bytes as f64/self.batch.max(1)as f64,"hbm_states":hmap,"hbm_state_sum":self.hstates.iter().sum::<u64>(),"cores":cores,"budget":physical,"actual_storage_peak_bytes":self.actual_storage_peak,"bindings":self.binding,"next_bindings":self.next_bindings,"context_switches":self.context_switches,"Z_live_peak_bytes":self.z_arena.peak,"U_live_peak_bytes":self.u_arena.peak,"tail_guard_fallbacks":self.tail_fallbacks,"work_steal_attempts":self.steal_attempts,"work_steal_successes":self.steals,"rank_factor_bytes":self.rank_factor_bytes,"lambda_before":self.lambda,"lambda_after":self.lambda_next(),"lambda0":self.cfg.raw.get("lambda0").and_then(Value::as_f64).unwrap_or(1.0),"quota_updates":self.quota_updates,"quota_denials":self.pool_quota_denials,"pool_capacity_denials":self.pool_capacity_denials,"latency_ewma_cycles":self.lat_ewma,"prediction_mean_absolute_error_cycles":self.pred_error/self.experts.len().max(1)as f64,"prediction_worst_underestimate_cycles":self.pred_worst,"cross_core_bytes":self.copy_bytes,"offload_helper_cycles":self.helper_cycles,"completion_order":self.completion,"drained":true,"invariants":{"unique_task_owner":true,"requests_drained":true,"pool_references_released":true,"unique_weight_bytes_exact":true,"main_useful_macs_exact":true,"stall_states_exclusive":true,"rank_capacity_checked_by_plan":true,"storage_fits":true,"combine_rmw_atomic":true,"k_order_preserved":true,"gather_addresses_validated":true},"trace":self.trace,"limitations":["HBM has fixed response latency and aggregate bandwidth, not native channel/row timing","optional unpacked numeric payload executes on the actual timed issue/vector/combine stream; raw MX bit-unpacking is covered by compiler numerical tests","dynamic rank allocation requires calibrated per-expert tail-energy tables; missing rows are rejected","offload U/B helper waits for current cold expert to finish; input and per-column output copies are charged before consumers execute"]});
        let extra=json!({"all_onchip_traffic_bytes":onchip,"legacy_onchip_subset_bytes":legacy_onchip,"onchip_movement_breakdown_bytes":movement,"onchip_traffic_definition":"sum of SRAM/RF endpoint reads/writes, ingress FIFO two endpoints per accepted drained byte, WOR/XOR array broadcast reads, and explicit cross-core wire copies; individual matrix PE wiring is excluded","joint_proposals":self.joint_proposals,"joint_comparisons":self.joint_comparisons,"joint_cancellations":self.joint_cancellations,"routed_rank_factor_bytes":self.routed_rank_factor_bytes,"shared_rank_factor_bytes":self.shared_rank_factor_bytes,"rank_budget_scope":s(&self.cfg.raw,"rank_budget_scope","routed"),"quota_progress_borrows":self.quota_progress_borrows,"pool_allocation_mode":if self.cfg.byte_pool{"shared byte-addressed pool"}else{"same-capacity fixed 4KiB slots with static private partitions"},"pool_reservation_granularity_bytes":if self.cfg.byte_pool{32}else{4096},"pool_partition_bytes_per_core":(0..self.cores.len()).map(|c|{let (lo,hi)=self.pool_partition(c);if self.cfg.byte_pool{self.cfg.pool}else{hi-lo}}).collect::<Vec<_>>()});
        for (k,v) in extra.as_object().unwrap(){report[k]=v.clone();}
        Ok(report)
    }
}
pub fn run(input: &Value) -> Result<Value, String> {
    Engine::new(input)?.run()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn input(lanes: Vec<usize>) -> Value {
        json!({"workload":{"id":"small","batch":4,"hidden":512,"top_k":1,"experts":[{"id":0,"Me":4,"H":512,"F":16,"is_shared":true},{"id":1,"Me":1,"H":512,"F":16,"is_shared":false}]},"config":{"lanes":lanes,"ranks":{"shared":[8,8,8],"routed":[8,8,8]},"rank_lanes":8}})
    }
    #[test]
    fn deterministic_and_drained() {
        let a = run(&input(vec![4, 2])).unwrap();
        let b = run(&input(vec![4, 2])).unwrap();
        assert_eq!(a, b);
        assert_eq!(a["drained"], true);
        assert_eq!(a["cycles"], a["hbm_state_sum"]);
        for c in a["cores"].as_array().unwrap() {
            assert_eq!(c["state_sum"], a["cycles"]);
        }
        let movement_sum=a["onchip_movement_breakdown_bytes"].as_object().unwrap().values().map(|v|v.as_u64().unwrap()).sum::<u64>();
        assert_eq!(movement_sum,a["all_onchip_traffic_bytes"].as_u64().unwrap());
        assert_eq!(a["all_onchip_traffic_bytes"],a["onchip_traffic_bytes"]);
    }
    #[test]
    fn all_orgs_and_precisions() {
        for lanes in [
            vec![6],
            vec![3, 3],
            vec![4, 2],
            vec![8],
            vec![4, 4],
            vec![6, 2],
            vec![5, 3],
        ] {
            for precision in ["P0", "P1", "P2"] {
                let mut x = input(lanes.clone());
                x["config"]["precision"] = json!(precision);
                let r = run(&x).unwrap();
                assert_eq!(r["invariants"]["main_useful_macs_exact"], true);
            }
        }
    }
    #[test]
    fn only_byte_compression_preserves_native_granularity() {
        let a = run(&input(vec![4, 2])).unwrap();
        assert_eq!(a["weight_bytes"].as_u64().unwrap() % 32, 0);
        assert!(a["pool_peak_bytes"].as_u64().unwrap() <= 65536);
    }
    #[test]
    fn two_contexts_alternate_without_extra_accumulator() {
        let q=json!({"workload":{"id":"contexts","batch":16,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":16,"H":544,"F":160,"is_shared":true},{"id":1,"Me":2,"H":544,"F":160,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P0","comp_mode":"none","rank_lanes":0,"placement":"fifo","contexts_per_core":2,"record_trace":true,"context_switch_policy":"alternating","wor_tiles":[4],"max_cycles":100000}});
        let r=run(&q).unwrap();assert!(r["context_switches"].as_u64().unwrap()>0);
        let core=&r["cores"][0];assert!(core["accumulator_live_peak_bytes"].as_u64().unwrap()<=core["accumulator_capacity_bytes"].as_u64().unwrap());assert_eq!(r["completion_order"].as_array().unwrap().len(),2);
    }
    #[test]
    fn streamed_down_uses_partial_combine_and_charged_ud_backing() {
        let q=json!({"workload":{"id":"streamed","batch":4,"hidden":544,"top_k":1,"experts":[{"id":1,"Me":4,"H":544,"F":640,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P2","factor_a":"mxint4","factor_b":"bf16","rank_lanes":4,"ranks":{"routed":[8,8,16],"shared":[8,8,16]},"z_mode":"streamed","record_trace":true,"trace_limit":100000,"max_cycles":100000}});
        let r=run(&q).unwrap();let tr=r["trace"].as_array().unwrap();
        let partial=tr.iter().filter(|v|v["event"]=="combine_commit"&&v["partial"]==true).count();
        assert_eq!(partial,ceil(544,32)*(ceil(640,512)+ceil(16-4,4)));
        assert!(r["cores"][0]["u_fp32_rmw_bytes"].as_u64().unwrap()>0);assert!(r["cores"][0]["u_bf16_store_bytes"].as_u64().unwrap()>0);
        assert_eq!(r["padding_main_macs"].as_u64().unwrap()+r["useful_macs"].as_u64().unwrap(),r["issued_main_macs"].as_u64().unwrap());
        assert_eq!(r["padding_aux_macs"].as_u64().unwrap()+r["useful_aux_macs"].as_u64().unwrap(),r["issued_aux_macs"].as_u64().unwrap());
    }
    #[test]
    fn full_z_down_finishes_tail_before_each_bounded_band_combine() {
        let q=json!({"workload":{"id":"full","batch":4,"hidden":544,"top_k":1,"experts":[{"id":1,"Me":4,"H":544,"F":160,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P2","rank_lanes":4,"ranks":{"routed":[8,8,16],"shared":[8,8,16]},"z_mode":"full","record_trace":true,"trace_limit":100000,"max_cycles":100000}});
        let r=run(&q).unwrap();let mut pending=BTreeMap::new();let mut combined=0;
        for ev in r["trace"].as_array().unwrap(){if ev["event"]=="issue"&&ev["projection"]==2 {pending.insert(ev["n"].as_u64().unwrap(),true);}else if ev["event"]=="combine_commit" {assert_eq!(ev["partial"],false);assert!(pending.len()<=8);pending.clear();combined+=1;}}
        assert_eq!(combined,ceil(544,32));
    }
    #[test]
    fn packed_u_rank_mapping_matches_written_and_read_words() {
        for z in ["full","streamed"] {
            let q=json!({"workload":{"id":"u-layout","batch":3,"hidden":544,"top_k":1,"experts":[{"id":1,"Me":3,"H":544,"F":640,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P2","rank_lanes":4,"ranks":{"routed":[32,24,32],"shared":[32,24,32]},"z_mode":z,"ideal_onchip":true}});
            let mut e=Engine::new(&q).unwrap();e.bind();assert!(e.activate_context(0));
            for p in 0..3{let r=e.tasks[0].plan.ranks[p];let positions:std::collections::BTreeSet<_>=(0..r).map(|j|e.rank_position(0,p,j)).collect();assert_eq!(positions,(0..r).collect());}
            let written:std::collections::BTreeSet<_>=e.u_store_spans(0,3,0,56).into_iter().chain(e.u_store_spans(0,4,0,32)).map(|v|v.0).collect();
            let groups:Vec<_>=e.tasks[0].plan.actions.iter().filter_map(|a|if let Action::Group(g)=a{if g.kind==Kind::Main{Some(g.clone())}else{None}}else{None}).collect();
            for g in groups{
                e.load_rank_cache(0,0,&g);
                let wanted_mask=e.tasks[0].plan.tiles[g.first_tile..g.first_tile+g.tile_count].iter().filter(|v|v.ranks>0).fold(0usize,|m,v|m|(1<<v.projection));
                if wanted_mask==0{continue;}let cache=e.cores[0].rank_caches.iter().find(|v|v.tag==Some((0,g.kseg,0,wanted_mask))).unwrap();
                let mask=cache.tag.unwrap().3;let mut expected=std::collections::BTreeSet::new();let base=e.u_base(0);
                for row in 0..3{for p in 0..3{if mask&(1<<p)==0{continue;}let r=e.tasks[0].plan.ranks[p];let fused=r.min(4*if p==2&&z=="streamed"{1}else{2});
                    for j in 0..fused{if !(p==2&&z=="streamed")&&j%2!=g.kseg{continue;}let position=if p==2&&z=="streamed"{j}else{[0,4,1,5,2,6,3,7][j]};let off=e.tasks[0].plan.ranks[..p].iter().sum::<usize>();expected.insert((base+row*88*2+(off+position)*2)/16*16);}
                }}
                assert_eq!(cache.spans.iter().map(|v|v.0).collect::<std::collections::BTreeSet<_>>(),expected);
                for cache in &e.cores[0].rank_caches{for &(addr,len) in &cache.spans{assert_eq!(len,16);assert!(written.contains(&addr));}}
            }
            if z=="streamed"{for j in 0..32{assert_eq!(e.rank_position(0,2,j),j);}}
        }
    }
    #[test]
    fn group_resume_requires_all_weights_before_parking_quota_owner() {
        let q=json!({"workload":{"id":"quota-resume","batch":3,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":3,"H":544,"F":640,"is_shared":true},{"id":7,"Me":2,"H":544,"F":640,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P2","rank_lanes":4,"ranks":{"routed":[8,8,16],"shared":[8,8,16]},"z_mode":"streamed","context_switch_policy":"starved","max_cycles":100000}});
        let r=run(&q).unwrap();assert_eq!(r["completion_order"].as_array().unwrap().len(),2);assert_eq!(r["drained"],true);assert_eq!(r["invariants"]["pool_references_released"],true);
    }
    #[test]
    fn gate_score_alias_float_byte_lut_and_lambda_reset_are_charged() {
        let values=json!([100.,50.,0.,0.,0.]);let costs=json!([0.,10.,20.,30.,40.]);
        let q=json!({"workload":{"id":"lut","batch":1,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":1,"H":544,"F":32,"is_shared":false,"route_scores":[0.01]}]},"config":{"lanes":[6],"precision":"P2","rank_alloc":"gate_weighted","lambda0":1.,"lambda_before":0.01,"rank_budget_bytes":32,"rank_energy_table":{"0":{"gate":values,"up":values,"down":values},"factor_bytes_per_projection":{"gate":[costs],"up":[costs],"down":[costs]}}}});
        let r=run(&q).unwrap();assert_eq!(r["bindings"][0]["ranks"],json!([0,0,0]));assert_eq!(r["bindings"][0]["rank_selection_cycles"],16);assert_eq!(r["lambda_before"],0.01);assert_eq!(r["lambda_after"],1.0);assert_eq!(r["budget"]["rank_lut_bytes"],6186);assert!(r["budget"]["control_used_bytes"].as_u64().unwrap()<=16384);
    }
    #[test]
    fn steal_drains_prefetched_old_response_leases_and_counts_retake() {
        let q=json!({"workload":{"id":"steal","batch":8,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":8,"H":544,"F":160,"is_shared":false},{"id":1,"Me":2,"H":544,"F":32,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group","ws_group"],"precision":"P0","comp_mode":"none","control_costs":false,"placement":"fifo","contexts_per_core":2,"context_interleave":false,"record_trace":true,"max_cycles":100000}});
        let mut e=Engine::new(&q).unwrap();e.bind();e.tasks[0].predicted=1;e.cores[1].cur=Some(0);e.cores[1].next=Some(0);e.control_free=0;e.bind();e.cores[1].cur=None;e.cores[1].next=None;e.tasks[0].predicted=1000000;e.control_free=0;
        assert_eq!(e.cores[0].next,Some(1));let old=e.tasks[1].offset;
        for _ in 0..1000{let mut busy=vec![false;e.cfg.pool_banks];e.returns();e.land(&mut busy);e.admit();e.hbm();for c in &mut e.cores{c.stats.states[7]+=1;}e.now+=1;if e.tiles[old].sent>0{break;}}
        assert!(e.tiles[old].sent>0);assert!(e.tiles[old].landed<e.tiles[old].sent);e.work_steal();assert_eq!(e.steals,1);assert!(e.tiles[old].discarded);assert_ne!(e.tasks[1].offset,old);assert_eq!(e.tasks[1].core,1);
        let r=e.run().unwrap();assert!(r["work_steal_waste_bytes"].as_u64().unwrap()>0);assert_eq!(r["weight_bytes"].as_u64().unwrap(),r["native_expert_bytes"].as_u64().unwrap()+r["comp_padding_bytes"].as_u64().unwrap()+r["work_steal_waste_bytes"].as_u64().unwrap());assert_eq!(r["drained"],true);assert_eq!(r["dma_transactions_accepted"],r["dma_transactions_landed"]);
    }

    #[test]
    fn quota_off_is_current_group_only_and_fixed_policy_is_real() {
        let mut q=input(vec![6]);q["config"]["dataflow"]=json!(["ws_group"]);q["config"]["placement"]=json!("fifo");q["config"]["control_cost"]=json!(false);q["config"]["prefetch_quota"]=json!(false);
        let mut e=Engine::new(&q).unwrap();e.bind();e.tasks[0].predicted=1;e.bind();assert_eq!(e.cores[0].next,Some(1));e.admit();assert_eq!(e.tasks[1].admit,0);let g=e.group_at(0,0).unwrap();assert_eq!(e.tasks[0].admit,g.first_tile+g.tile_count);
        q["config"]["prefetch_quota"]=json!(true);q["config"]["quota_policy"]=json!("fixed_one_tile");let mut e=Engine::new(&q).unwrap();e.bind();e.tasks[0].predicted=1;e.bind();e.admit();assert_eq!(e.tasks[1].admit,1);
    }
    #[test]
    fn nonpipelined_reads_wait_real_accumulator_completion() {
        let mut q=input(vec![6]);q["config"]["pipeline_supply"]=json!(false);q["config"]["record_trace"]=json!(true);q["config"]["dataflow"]=json!(["ws_group"]);let r=run(&q).unwrap();let trace=r["trace"].as_array().unwrap();let starts=trace.iter().filter(|v|v["event"]=="wor_read_start").collect::<Vec<_>>();assert!(!starts.is_empty());for v in starts{assert!(v["cycle"].as_u64().unwrap()>=v["previous_acc_commit"].as_u64().unwrap());}
        assert_eq!(r["budget"]["pipeline_supply_enabled"],false);assert_eq!(r["drained"],true);
    }
    #[test]
    fn deferred_silu_has_real_fp32_spill_write_and_read_in_fixed_arena() {
        for flow in ["ws_group","switchable"] {
            let mut q=input(vec![6]);q["config"]["dataflow"]=json!([flow]);q["config"]["inline_silu"]=json!(false);q["config"]["record_trace"]=json!(true);let r=run(&q).unwrap();let c=&r["cores"][0];assert!(c["silu_spill_read_bytes"].as_u64().unwrap()>0);if flow=="ws_group"{assert!(c["silu_spill_write_bytes"].as_u64().unwrap()>0);assert_eq!(c["silu_copy_source_read_bytes"],c["silu_spill_write_bytes"]);assert_eq!(c["silu_spill_read_bytes"],c["silu_spill_write_bytes"]);}assert!(c["private_accumulator_address_peak_bytes"].as_u64().unwrap()<=c["accumulator_capacity_bytes"].as_u64().unwrap());assert_eq!(r["budget"]["accumulator_source_and_rmw_share_port"],true);
        }
    }
    #[test]
    fn joint_snapshot_keeps_pair_and_differs_from_greedy_finish() {
        let mut q=input(vec![3,3]);q["config"]["placement"]=json!("joint");q["config"]["control_cost"]=json!(false);q["config"]["dataflow"]=json!(["switchable","switchable"]);q["workload"]["experts"][0]["is_shared"]=json!(false);q["workload"]["experts"][0]["Me"]=json!(1);q["workload"]["experts"][1]["Me"]=json!(4);
        let mut joint=Engine::new(&q).unwrap();joint.bind();assert_eq!(joint.joint_pending.len(),2);assert_eq!(joint.joint_pending[0].0,1);joint.bind();assert_eq!(joint.tasks[0].expert,1);joint.bind();assert_eq!(joint.tasks.len(),2);assert_ne!(joint.tasks[0].core,joint.tasks[1].core);assert!(joint.joint_comparisons>0);
        q["config"]["placement"]=json!("earliest_finish");let mut eft=Engine::new(&q).unwrap();eft.bind();assert_eq!(eft.tasks[0].expert,0);
    }
    #[test]
    fn parked_partial_wor_is_not_a_free_operand_context() {
        let mut q=input(vec![4,2]);q["config"]["dataflow"]=json!(["switchable","switchable"]);q["config"]["placement"]=json!("fifo");let mut e=Engine::new(&q).unwrap();e.bind();let t=e.cores[0].cur.unwrap();e.control_free=0;e.tasks[t].predicted=1;e.bind();let other=e.cores[0].next.or(e.cores[1].cur).unwrap();e.cores[0].next=Some(other);e.tasks[other].core=0;let i=e.tasks[t].offset;e.tiles[i].resident=true;e.tasks[t].resident_tiles=1;e.tiles[i].loaded=false;e.alternate_context(t);assert_eq!(e.cores[0].cur,Some(t));
    }

    #[test]
    fn two_context_progress_under_undersized_head_quota_and_off_modes() {
        for lanes in [vec![6],vec![3,3],vec![4,2]]{for precision in ["P0","P1"]{for mode in ["default","quota_off","pipeline_off","fixed_one_tile"]{
            let nc=lanes.len();let mut q=json!({"workload":{"id":"bounded-next","batch":16,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":16,"H":544,"F":160,"is_shared":true},{"id":1,"Me":2,"H":544,"F":160,"is_shared":false}]},"config":{"lanes":lanes,"dataflow":vec!["ws_group";nc],"wor_tiles":vec![8;nc],"precision":precision,"ranks":{"shared":[8,8,8],"routed":[8,8,8]},"rank_lanes":8,"placement":"fifo","contexts_per_core":2,"max_cycles":100000}});
            if mode=="quota_off"{q["config"]["prefetch_quota"]=json!(false);}if mode=="pipeline_off"{q["config"]["pipeline_supply"]=json!(false);}if mode=="fixed_one_tile"{q["config"]["quota_policy"]=json!("fixed_one_tile");}
            let a=run(&q).unwrap_or_else(|e|panic!("{precision} {mode} {q}: {e}"));let b=run(&q).unwrap();assert_eq!(a,b);assert_eq!(a["drained"],true);assert_eq!(a["dma_transactions_accepted"],a["dma_transactions_landed"]);
        }}}
    }
    #[test]
    fn rank_feedback_uses_routed_budget_and_keeps_shared_fixed_cost_separate() {
        let mut q=input(vec![6]);let mut e=Engine::new(&q).unwrap();e.cfg.rank_alloc="gate_weighted".into();e.cfg.raw["lambda0"]=json!(1.);e.cfg.raw["rank_budget_bytes"]=json!(100);e.lambda=1.;e.rank_factor_bytes=500;e.shared_rank_factor_bytes=400;e.routed_rank_factor_bytes=100;assert_eq!(e.lambda_next(),1.);
        q["config"]["rank_budget_scope"]=json!("all");e.cfg.raw["rank_budget_scope"]=json!("all");assert!(e.lambda_next()>1.);
    }
    #[test]
    fn contracted_quota_cannot_block_current_behind_held_next_head() {
        let mut q=input(vec![6]);q["config"]["placement"]=json!("fifo");q["config"]["control_cost"]=json!(false);q["config"]["dataflow"]=json!(["ws_group"]);let mut e=Engine::new(&q).unwrap();e.bind();e.tasks[0].predicted=1;e.bind();
        let i=e.tasks[1].offset;let b=e.tiles[i].spec.bytes;let addr=e.pool.alloc(b).unwrap();e.tiles[i].addr=Some(addr);e.tiles[i].reserved_bytes=b;e.tasks[1].admit=1;e.tasks[1].bytes_live=b;e.cores[0].live=b;e.cores[0].quota=b/2;let ng=e.group_at(1,0).unwrap();assert!(ng.tile_count>1);
        let g=e.group_at(0,0).unwrap();e.admit();assert_eq!(e.tasks[0].admit,g.first_tile+g.tile_count);assert!(e.quota_progress_borrows>0);assert_eq!(e.tasks[1].admit,1);assert!(e.pool.used<=e.cfg.pool);
        let r=e.run().unwrap();assert_eq!(r["drained"],true);assert_eq!(r["dma_transactions_accepted"],r["dma_transactions_landed"]);assert!(r["quota_progress_borrows"].as_u64().unwrap()>0);
    }
    #[test]
    fn helper_delta_has_finite_source_and_owner_rmw_inside_installed_arena() {
        let mut q=input(vec![4,2]);q["config"]["comp_mode"]=json!("offload");q["config"]["precision"]=json!("P1");q["config"]["record_trace"]=json!(true);q["config"]["trace_limit"]=json!(100000);q["config"]["dataflow"]=json!(["ws_group","ws_group"]);
        let r=run(&q).unwrap();let copies=r["trace"].as_array().unwrap().iter().filter(|v|v["event"]=="helper_delta_return").collect::<Vec<_>>();assert!(!copies.is_empty());
        for ev in copies{assert_eq!(ev["source_read_bytes"],ev["bytes"]);assert_eq!(ev["owner_rmw_bytes"].as_u64().unwrap(),2*ev["bytes"].as_u64().unwrap());assert_eq!(ev["helper_delta_cleared"],true);}
        assert!(r["cores"][1]["consumer_private_source_read_bytes"].as_u64().unwrap()>0);for c in r["cores"].as_array().unwrap(){assert!(c["private_accumulator_address_peak_bytes"].as_u64().unwrap()<=c["accumulator_capacity_bytes"].as_u64().unwrap());}assert_eq!(r["drained"],true);
    }
    #[test]
    fn fixed_slot_pool_is_same_capacity_with_real_private_fragmentation(){
        for lanes in [vec![6],vec![3,3],vec![4,2]]{for precision in ["P0","P1","P2"]{let nc=lanes.len();let mut q=input(lanes.clone());q["config"]["precision"]=json!(precision);q["config"]["byte_pool"]=json!(false);q["config"]["record_trace"]=json!(true);q["config"]["dataflow"]=json!(vec!["ws_group";nc]);let r=run(&q).unwrap();assert_eq!(r["drained"],true);assert!(r["pool_peak_bytes"].as_u64().unwrap()<=65536);let mut short=false;
            for ev in r["trace"].as_array().unwrap().iter().filter(|v|v["event"]=="reserve"){let a=ev["addr"].as_u64().unwrap()as usize;let reserved=ev["reserved_bytes"].as_u64().unwrap()as usize;let payload=ev["bytes"].as_u64().unwrap()as usize;let c=ev["owner"].as_u64().unwrap()as usize;let part=65536/nc;assert_eq!(reserved,align(payload,4096));assert!(a>=c*part&&a+reserved<=(c+1)*part);short|=reserved>payload;}assert!(short);
            q["config"]["byte_pool"]=json!(true);let b=run(&q).unwrap();assert_eq!(b["weight_bytes"],r["weight_bytes"]);assert_eq!(b["unique_weight_bytes"],r["unique_weight_bytes"]);
        }}
    }
    #[test]
    fn disabled_w_reuse_refills_after_each_real_m_block(){
        for precision in ["P0","P1","P2"]{let mut q=input(vec![4,2]);q["workload"]["batch"]=json!(8);q["workload"]["experts"][0]["Me"]=json!(8);q["config"]["precision"]=json!(precision);q["config"]["dataflow"]=json!(["ws_group","ws_group"]);q["config"]["w_reuse"]=json!(false);q["config"]["placement"]=json!("fifo");q["config"]["record_trace"]=json!(true);q["config"]["trace_limit"]=json!(100000);let r=run(&q).unwrap();assert_eq!(r["drained"],true);let mut event=0;let mut issued0=0;
            for v in r["trace"].as_array().unwrap().iter().filter(|v|v["tile"]==0){let name=v["event"].as_str().unwrap();match name{"wor_read_start" if v["m_block"]==0=>{assert_eq!(event,0);event=1;},"wor_loaded" if v["m_block"]==0=>{assert_eq!(event,1);assert_eq!(v["pool_released"],false);event=2;},"issue" if v["m_block"]==0=>{assert_eq!(event,2);issued0=v["cycle"].as_u64().unwrap();event=3;},"wor_retire_m_block"=>{assert_eq!(event,3);event=4;},"wor_read_start" if v["m_block"]==1=>{assert_eq!(event,4);assert!(v["cycle"].as_u64().unwrap()>issued0);event=5;},"wor_loaded" if v["m_block"]==1=>{assert_eq!(event,5);assert_eq!(v["pool_released"],true);event=6;},"issue" if v["m_block"]==1=>{assert_eq!(event,6);event=7;},_=>{}}}assert_eq!(event,7);
        }
    }
    #[test]
    fn forward_zero_all_supply_switches_off_drains_every_organization_and_precision(){
        for lanes in [vec![6],vec![3,3],vec![4,2]]{for precision in ["P0","P1","P2"]{let nc=lanes.len();let mut q=json!({"workload":{"id":"forward-zero","batch":16,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":16,"H":544,"F":160,"is_shared":true},{"id":1,"Me":2,"H":544,"F":160,"is_shared":false}]},"config":{"lanes":lanes,"dataflow":vec!["ws_group";nc],"wor_tiles":vec![8;nc],"precision":precision,"ranks":{"shared":[8,8,8],"routed":[8,8,8]},"rank_lanes":8,"byte_pool":false,"prefetch_quota":false,"pipeline_supply":false,"x_reuse":false,"w_reuse":false,"wide_ports":false,"inline_silu":false,"credit_release":"landing","credits":256,"placement":"fifo","contexts_per_core":2,"max_cycles":100000}});let a=run(&q).unwrap_or_else(|s|panic!("{q}: {s}"));let b=run(&q).unwrap();assert_eq!(a,b);assert_eq!(a["drained"],true);assert_eq!(a["dma_transactions_accepted"],a["dma_transactions_landed"]);assert!(a["pool_peak_bytes"].as_u64().unwrap()<=65536);}}
    }
    #[test]
    fn private_next_head_is_atomic_when_static_slots_are_insufficient(){
        let mut q=input(vec![4,2]);q["config"]["placement"]=json!("fifo");q["config"]["control_cost"]=json!(false);q["config"]["dataflow"]=json!(["ws_group","ws_group"]);q["config"]["byte_pool"]=json!(false);let mut e=Engine::new(&q).unwrap();e.bind();let t=e.cores[0].cur.unwrap();e.tasks[t].predicted=1;e.bind();let next=e.cores[0].next.or(e.cores[1].cur).unwrap();e.cores[1].cur=None;e.cores[0].next=Some(next);e.tasks[next].core=0;e.cores[0].quota=32768;let cg=e.group_at(t,0).unwrap();for i in cg.first_tile..cg.first_tile+cg.tile_count{e.tiles[e.tasks[t].offset+i].loaded=true;}
        let ng=e.group_at(next,0).unwrap();let need=ng.tile_count*4096;assert!(need>4096);let fake=e.pool.alloc_partition(32768-need+4096,0,32768).unwrap();e.admit();assert_eq!(e.tasks[next].admit,0);e.pool.release(fake,32768-need+4096);
    }
    #[test]
    fn drained_helper_does_not_pin_full_stream_accumulator() {
        let q=json!({"workload":{"id":"helper-cold-full-arena","batch":16,"hidden":544,"top_k":1,"experts":[{"id":-1,"Me":16,"H":544,"F":1280,"is_shared":true},{"id":1,"Me":1,"H":544,"F":640,"is_shared":false},{"id":2,"Me":4,"H":544,"F":640,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group","is_stream"],"precision":"P1","comp_mode":"offload","comp_equal_bytes":false,"rank_lanes":8,"ranks":{"shared":[8,8,8],"routed":[8,8,8]},"record_trace":true,"trace_limit":100000,"max_cycles":1000000}});
        let a=run(&q).unwrap();let b=run(&q).unwrap();assert_eq!(a,b);
        assert_eq!(a["drained"],true);assert_eq!(a["dma_transactions_accepted"],a["dma_transactions_landed"]);
        assert!(a["trace"].as_array().unwrap().iter().any(|v|v["event"]=="helper_acc_retire"));
        for c in a["cores"].as_array().unwrap(){assert!(c["private_accumulator_address_peak_bytes"].as_u64().unwrap()<=c["accumulator_capacity_bytes"].as_u64().unwrap());}
    }
    #[test]
    fn carried_lambda_survives_exact_json_roundtrip() {
        let value=6.449598548152563e-8_f64;
        let encoded=serde_json::to_string(&json!({"lambda_before":value})).unwrap();
        let decoded:Value=serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded["lambda_before"].as_f64().unwrap().to_bits(),value.to_bits());
    }
    #[test]
    fn illegal_precision_is_error() {
        let mut x = input(vec![4, 2]);
        x["config"]["comp_mode"] = json!("kext");
        assert!(run(&x).is_err());
    }
}
