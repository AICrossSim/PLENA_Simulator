//! Nonphysical projection oracle: resident operands, zero memory/control cost.
//! Reuses physical tiling; II=1, dot latency=21, 8 output contexts per core,
//! ascending K and N-group retirement remain. Not full-system timing.
use super::*;

pub(super) fn run(input: &Value) -> Value {
    let lanes: Vec<usize> = serde_json::from_value(input["lanes"].clone()).unwrap();
    let ms: Vec<usize> = serde_json::from_value(input["Me"].clone()).unwrap();
    let owners: Vec<usize> = serde_json::from_value(input["owners"].clone()).unwrap();
    let n = input["N"].as_u64().unwrap() as usize;
    let k = input["K"].as_u64().unwrap() as usize;
    let group = input["group"].as_u64().unwrap() as usize;
    assert_eq!(ms.len(), owners.len());
    let mut ends = vec![0u64; lanes.len()];
    let mut issues = vec![0u64; lanes.len()];
    let mut useful = 0u64;
    let mut allocated = 0u64;
    let mut tasks = vec![];
    for (e, &m) in ms.iter().enumerate() {
        let c = owners[e];
        assert!(c < lanes.len());
        let start = ends[c];
        let mut t = start;
        let mut inflight: Vec<u64> = vec![];
        let mut committed = BTreeMap::<(usize, usize), u64>::new();
        let mut previous_n = None;
        for g in build_groups(
            Projection {
                m,
                n,
                k,
                n_start: 0,
            },
            lanes[c],
            group,
        )
        .unwrap()
        {
            if previous_n != Some(g.n_start) {
                t = t.max(inflight.iter().copied().max().unwrap_or(t));
                inflight.clear();
                committed.clear();
                previous_n = Some(g.n_start);
            }
            for i in g.issues {
                t = t.max(*committed.get(&(i.m_start, i.n_start)).unwrap_or(&0));
                inflight.retain(|&done| done > t);
                if inflight.len() >= 8 {
                    t = t.max(*inflight.iter().min().unwrap());
                    inflight.retain(|&done| done > t);
                }
                let done = t + 21; // one operand issue cycle plus current dot tail 20.
                committed.insert((i.m_start, i.n_start), done);
                inflight.push(done);
                useful += i.useful_macs();
                allocated += i.issued_macs(lanes[c]);
                issues[c] += 1;
                t += 1;
            }
        }
        ends[c] = t.max(inflight.iter().copied().max().unwrap_or(t));
        tasks.push(json!({"expert":e,"core":c,"Me":m,"M_blocks":ceil(m,lanes[c]),"start":start,"done":ends[c]}));
    }
    let cycles = *ends.iter().max().unwrap();
    json!({"scope":"nonphysical compute-only projection oracle; no HBM/SRAM/control costs",
        "cycles":cycles,"useful_macs":useful,"issued_macs":allocated,"padding_macs":allocated-useful,
        "spatial_utilization":useful as f64/allocated as f64,"issues_per_core":issues,
        "core_finish":ends,"core_idle_after_finish":ends.iter().map(|&t|cycles-t).collect::<Vec<_>>(),
        "initiation_interval":1,"dot_completion_latency":21,"result_contexts_per_core":8,
        "N_group_retirement":true,"tasks":tasks})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tails_and_overlap_are_counted_separately() {
        let r = run(&json!({"lanes":[3,3],"Me":[4,2],"owners":[0,1],"N":128,"K":512,"group":4}));
        assert_eq!(r["useful_macs"], 393216);
        assert_eq!(r["padding_macs"], 196608);
        assert!(r["cycles"].as_u64().unwrap() < 64 * 21);
        let r = run(&json!({"lanes":[4,2],"Me":[4,2],"owners":[0,1],"N":128,"K":512,"group":4}));
        assert_eq!(r["padding_macs"], 0);
    }
}
