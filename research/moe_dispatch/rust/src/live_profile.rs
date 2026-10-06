//! Observers only. Half-open intervals are unioned; concurrent issues are not summed.
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn merge(mut xs: Vec<(u64,u64)>) -> Vec<(u64,u64)> {
    xs.sort_unstable();
    let mut out: Vec<(u64,u64)> = Vec::new();
    for (a,b) in xs {
        assert!(a <= b);
        if a == b { continue; }
        if let Some(last) = out.last_mut().filter(|last| a <= last.1) {
            last.1 = last.1.max(b);
        } else { out.push((a,b)); }
    }
    out
}
fn length(xs: &[(u64,u64)]) -> u64 { xs.iter().map(|(a,b)| b-a).sum() }

pub(super) struct Profile {
    mac: Vec<Vec<(u64,u64)>>,
    pipeline: Vec<Vec<(u64,u64)>>,
    requests: BTreeMap<u64,(u64,usize)>,
    accepted: u64,
    returned: u64,
    first_request: Option<u64>,
    last_return: Option<u64>,
    latency_sum: u128,
    latency_max: u64,
    latency_hist: BTreeMap<u64,u64>,
    pub outstanding_cycles: u64,
    pub native_backpressure_cycles: u64,
}
impl Profile {
    pub fn new(n: usize) -> Self { Self { mac: vec![vec![];n], pipeline:vec![vec![];n],
        requests:BTreeMap::new(), accepted:0, returned:0, first_request:None,last_return:None,
        latency_sum:0,latency_max:0,latency_hist:BTreeMap::new(),outstanding_cycles:0,
        native_backpressure_cycles:0 } }
    pub fn issue(&mut self,c:usize,start:u64,operand_ready:u64,end:u64) {
        assert!(start <= operand_ready && operand_ready <= end);
        self.mac[c].push((operand_ready,end)); self.pipeline[c].push((start,end));
    }
    pub fn accept(&mut self,serial:u64,c:usize,now:u64) {
        assert!(self.requests.insert(serial,(now,c)).is_none());
        self.first_request.get_or_insert(now); self.accepted+=1;
    }
    pub fn returned(&mut self,serial:u64,now:u64) {
        let (start,_) = self.requests.remove(&serial).expect("return without accepted request");
        let latency=now-start;
        self.latency_sum+=latency as u128; self.latency_max=self.latency_max.max(latency);
        *self.latency_hist.entry(latency).or_default()+=1;
        self.last_return=Some(now); self.returned+=1;
    }
    pub fn report(&self,wall:u64,macs:&[u64],weight_bytes:u64) -> Value {
        assert!(self.requests.is_empty()); assert_eq!(self.accepted,self.returned);
        assert_eq!(self.accepted*32,weight_bytes);
        let mac=merge(self.mac.iter().flatten().copied().collect());
        let pipeline=merge(self.pipeline.iter().flatten().copied().collect());
        let compute=length(&mac); let pipe=length(&pipeline);
        let first=self.first_request.unwrap_or(0); let last=self.last_return.unwrap_or(first);
        let fetch=last-first;
        let overlap:u64=mac.iter().map(|(a,b)| b.min(&last).saturating_sub(*a.max(&first))).sum();
        let mut n=0; let mut p95=0;
        for (&lat,&count) in &self.latency_hist { n+=count; if n*100 >= self.returned*95 { p95=lat;break; } }
        json!({"schema":"live_timing_profile_v1", "cycle_ns":1,
            "interval_convention":"[start,end); merge across tiles and cores, never sum overlaps",
            "fetch_boundary":"first accepted weight request to last memory return callback; includes gaps; excludes final landing write",
            "compute_boundary":"operand-ready to Dot completion (configured arithmetic pipeline latency); union across all issues/cores",
            "pipeline_boundary":"issue to Dot completion; includes operand feeding",
            "first_weight_request_cycle":first,"last_weight_return_cycle":last,
            "fetch_span_cycles":fetch,"mac_active_union_cycles":compute,
            "operand_and_mac_pipeline_union_cycles":pipe,"fetch_mac_overlap_cycles":overlap,
            "wall_without_mac_active_cycles":wall-compute,
            "memory_outstanding_cycles":self.outstanding_cycles,
            "memory_outstanding_definition":"at least one accepted weight read awaiting callback; NOT HBM command/data bus busy",
            "native_frontend_backpressure_cycles":self.native_backpressure_cycles,
            "weight_requests_accepted":self.accepted,"weight_callbacks_returned":self.returned,
            "request_latency_mean_cycles":self.latency_sum as f64 / self.returned.max(1) as f64,
            "request_latency_p95_cycles":p95,"request_latency_max_cycles":self.latency_max,
            "fetch_bandwidth_GBps":weight_bytes as f64 / fetch.max(1) as f64,
            "wall_weight_bandwidth_GBps":weight_bytes as f64 / wall.max(1) as f64,
            "compute_active_GFLOPs":2.0*macs.iter().sum::<u64>() as f64 / compute.max(1) as f64,
            "wall_GFLOPs":2.0*macs.iter().sum::<u64>() as f64 / wall.max(1) as f64,
            "cores":(0..macs.len()).map(|c| json!({"core":c,
                "mac_active_union_cycles":length(&merge(self.mac[c].clone())),
                "operand_and_mac_pipeline_union_cycles":length(&merge(self.pipeline[c].clone())),
                "useful_macs":macs[c]})).collect::<Vec<_>>() })
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn union_does_not_double_count_concurrent_tiles_or_cores() {
        assert_eq!(merge(vec![(10,20),(0,12),(30,40),(40,45),(6,8)]),vec![(0,20),(30,45)]);
        let mut p=Profile::new(2);
        p.issue(0,0,10,30);p.issue(0,20,25,45);p.issue(1,0,15,35);
        p.accept(0,0,0);p.returned(0,50);
        let r=p.report(60,&[100,50],32);
        assert_eq!(r["mac_active_union_cycles"],35);
        assert_eq!(r["fetch_mac_overlap_cycles"],35);
        assert_eq!(r["cores"][0]["mac_active_union_cycles"],35);
        assert_eq!(r["cores"][1]["mac_active_union_cycles"],20);
    }
}
