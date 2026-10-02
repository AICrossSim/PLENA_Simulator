//! Capacity extension for previously infeasible legacy token windows.
//!
//! Each child executes the unchanged joint_v1 kernel and completely drains.
//! Global X/final Y/route records stay in the original private arenas. Child
//! X/Y addresses alias disjoint global row views; no free external storage or
//! DMA cache survives a chunk boundary. Repeated weights are real refetches.
use super::*;

fn u(v: &Value, k: &str) -> u64 { v[k].as_u64().unwrap_or_else(|| panic!("missing integer {k}")) }

/// Only additive counters/histograms use this helper; state/peaks are explicit.
fn add_tree(dst: &mut Value, src: &Value) {
    match src {
        Value::Number(n) if n.is_u64() => {
            *dst = json!(dst.as_u64().unwrap_or(0).checked_add(n.as_u64().unwrap()).unwrap());
        }
        Value::Object(map) => {
            if !dst.is_object() { *dst = json!({}); }
            for (key, value) in map { add_tree(&mut dst[key], value); }
        }
        Value::Array(values) => {
            if !dst.is_array() { *dst = json!([]); }
            let target = dst.as_array_mut().unwrap();
            while target.len() < values.len() { target.push(Value::Null); }
            for (a, b) in target.iter_mut().zip(values) { add_tree(a, b); }
        }
        _ => {}
    }
}

fn is_peak(key: &str) -> bool {
    key.contains("peak") || key.ends_with("_max") || key.ends_with("_max_cycles") || key == "max_bypass"
}

fn merge_stats(dst: &mut Value, src: &Value, offset: u64) {
    for (key, value) in src.as_object().unwrap() {
        if key == "done_cycle" {
            let done = value.as_u64().unwrap();
            dst[key] = json!(dst[key].as_u64().unwrap_or(0).max(if done == 0 { 0 } else { offset + done }));
        } else if is_peak(key) {
            dst[key] = json!(dst[key].as_u64().unwrap_or(0).max(value.as_u64().unwrap()));
        } else { add_tree(&mut dst[key], value); }
    }
}

fn verify_plan(parent: &Value, plan: &Value, cfg: &Config) {
    assert_eq!(plan["schema"], "plena_legacy_token_chunks_v1");
    assert_eq!(cfg.arch, "joint_v1");
    assert_eq!(cfg.split, "none", "outer chunks preserve whole-expert legacy kernels");
    assert!(cfg.fixed_assignment.is_empty(), "explicit full-window owners cannot be silently remapped");
    let persistent = plan["persistent_cores"].as_array().unwrap();
    assert_eq!(persistent.len(), cfg.lanes.len());
    let record = &plan["control_record"];
    let owner = u(record, "core") as usize;
    assert!(owner < persistent.len());
    assert_eq!(u(record, "bytes"), 32);
    assert!(u(record, "base") + 32 <= u(&persistent[owner], "control"));
    let original_account = &parent["engine_layout"]["control_accounting"];
    assert_eq!(u(record, "base"), original_account["state_bytes_per_core"][owner].as_u64().unwrap());
    for (index, core) in persistent.iter().enumerate() {
        assert_eq!(u(core, "capacity"), u(&parent["engine_layout"]["cores"][index], "capacity"));
        assert!(u(core, "reserved") <= u(core, "capacity"));
        for field in ["original_x", "combined_output", "route_state"] {
            let range = &core[field];
            assert!(u(range, "address_bytes") + u(range, "bytes") <= u(core, "reserved"));
        }
        let mut regions = ["original_x", "combined_output", "route_state"].map(|field| {
            (u(&core[field], "address_bytes"), u(&core[field], "bytes"))
        });
        regions.sort_unstable();
        assert!(regions.windows(2).all(|pair| pair[0].0 + pair[0].1 <= pair[1].0),
            "persistent input/output/routes overlap");
    }
    let original = parent["experts"].as_array().unwrap();
    let mut covered: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut next = 0;
    for part in plan["chunks"].as_array().unwrap() {
        let start = part["token_range"][0].as_u64().unwrap() as usize;
        let end = part["token_range"][1].as_u64().unwrap() as usize;
        assert_eq!(start, next);
        assert!(end > start && end - start <= u(plan, "chunk_size") as usize);
        let child = &part["workload"];
        assert_eq!(u(child, "batch") as usize, end - start);
        assert!(child.get("legacy_batch_execution").is_none());
        let account = &child["engine_layout"]["control_accounting"];
        assert_eq!(u(account,"total_state_bytes"),u(original_account,"total_state_bytes")+32);
        assert_eq!(u(account,"reserve_headroom_bytes")+32,u(original_account,"reserve_headroom_bytes"));
        for c in 0..persistent.len() {
            let extra = if c==owner {32} else {0};
            assert_eq!(account["state_bytes_per_core"][c].as_u64().unwrap(),original_account["state_bytes_per_core"][c].as_u64().unwrap()+extra);
            assert_eq!(account["headroom_bytes_per_core"][c].as_u64().unwrap()+extra,original_account["headroom_bytes_per_core"][c].as_u64().unwrap());
        }
        for expert in child["experts"].as_array().unwrap() {
            let id = expert["id"].as_i64().unwrap();
            let source = original.iter().find(|e| e["id"] == expert["id"]).expect("unknown chunk expert");
            for field in ["is_shared", "H", "F", "weights"] { assert_eq!(expert[field], source[field]); }
            let tokens = expert["token_indices"].as_array().unwrap();
            assert_eq!(tokens.len(), u(expert, "Me") as usize);
            for (row, token) in tokens.iter().enumerate() {
                let local = token.as_u64().unwrap() as usize;
                assert!(local < end - start);
                let global = start + local;
                let source_row = source["token_indices"].as_array().unwrap().iter()
                    .position(|t| t.as_u64().unwrap() as usize == global).expect("invented route");
                for field in ["route_slots", "route_scores"] {
                    if source.get(field).is_some() { assert_eq!(expert[field][row], source[field][source_row]); }
                }
                covered.entry(id).or_default().push(global);
            }
        }
        for (index, core) in child["engine_layout"]["cores"].as_array().unwrap().iter().enumerate() {
            assert_eq!(u(core, "capacity"), u(&persistent[index], "capacity"));
            assert!(u(core, "reserved") >= u(&persistent[index], "reserved"));
            for field in ["original_x", "combined_output"] {
                let global = &persistent[index][field];
                let view = &core["result_layout"][field];
                assert_eq!(u(view, "address_bytes"), u(global, "address_bytes") + start as u64 * u(global, "row_stride_bytes"));
                assert_eq!(u(view, "bytes"), (end - start) as u64 * u(global, "row_stride_bytes"));
            }
            assert!(u(core, "reserved") <= u(core, "capacity"));
        }
        next = end;
    }
    assert_eq!(next, u(parent, "batch") as usize);
    for source in original {
        let id = source["id"].as_i64().unwrap();
        let expected: Vec<_> = source["token_indices"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as usize).collect();
        assert_eq!(covered.get(&id), Some(&expected), "route order/population changed by chunking");
    }
}

/// One serial front-end pass. Descriptor rows are physically read from the
/// full resident router records, sequenced, then the 32B loop record is written.
/// This adds no X/Y payload move: those child views alias persistent rows.
fn setup(parent: &Value, plan: &Value, part: &Value, cfg: &Config) -> (u64, u64, u64, u64) {
    let record = &plan["control_record"];
    let owner = u(record, "core") as usize;
    let core = &plan["persistent_cores"][owner];
    let route_base = u(&core["route_state"], "address_bytes") as usize;
    let start = part["token_range"][0].as_u64().unwrap() as usize;
    let end = part["token_range"][1].as_u64().unwrap() as usize;
    let k = u(parent, "top_k") as usize;
    let mut spans = vec![(route_base + start * k * 16, (end - start) * k * 16)];
    let original = parent["experts"].as_array().unwrap();
    let experts = part["workload"]["experts"].as_array().unwrap();
    for e in experts {
        let index = original.iter().position(|x| x["id"] == e["id"]).unwrap();
        spans.push((route_base + u(parent, "batch") as usize * k * 16 + index * 64, 64));
    }
    let n_banks = parent["engine_layout"]["hardware"]["acc_banks"][owner].as_u64().unwrap() as usize;
    let mut banks = Banks::new(n_banks);
    let read_end = banks.access(0, &spans, true, false);
    let descriptor_cycles = if cfg.control_cost { ((end - start) * k + experts.len() + 4) as u64 } else { 0 };
    let before_write = if cfg.ideal_onchip { descriptor_cycles } else { read_end + descriptor_cycles };
    let write_end = banks.access(before_write, &[(u(record, "base") as usize, 32)], false, false);
    let cycles = if cfg.ideal_onchip { descriptor_cycles } else { write_end };
    let bytes = spans.iter().map(|(_, b)| *b as u64).sum::<u64>() + 32;
    (cycles, descriptor_cycles, banks.words, bytes)
}

pub(super) fn run(parent: &Value, cfg: Config) -> Value {
    let plan = &parent["legacy_batch_execution"];
    verify_plan(parent, plan, &cfg);
    let nc = cfg.lanes.len();
    let owner = u(&plan["control_record"], "core") as usize;
    let mut out = Value::Null;
    let mut parts = vec![];
    let mut cycles = 0u64;
    let mut setup_cycles = 0u64;
    let mut setup_bytes = 0u64;
    let mut profile = json!({"hbm_states":{},"core_states":vec![json!({});nc],
        "issue_interval_histograms":vec![json!({});nc],"issue_stage_samples":[],"stage_totals_count_operand_mac_rmw":{}});
    let sum_fields = ["useful_macs", "issued_macs", "weight_bytes", "dispatch_decisions", "deferrals",
        "dma_transactions_accepted", "dma_transactions_landed", "input_backpressure_cycles",
        "feedback_updates", "tail_partition_count", "late_bind_wait_cycles", "combine_tail_cycles"];
    let max_fields = ["credit_peak", "pending_window_peak"];
    let mut total_core = (0..nc).map(|c| json!({"m":cfg.lanes[c],
        "capacity":plan["persistent_cores"][c]["capacity"],"reserved_input_result_control":0,
        "stats":{},"weight_bank_words":0,"x_bank_words":0,"workspace_bank_words":0})).collect::<Vec<_>>();
    let mut sums = json!({});
    let mut audits = vec![];
    let mut traces = vec![];
    for (index, part) in plan["chunks"].as_array().unwrap().iter().enumerate() {
        let (setup_time, control_time, setup_words, bytes) = setup(parent, plan, part, &cfg);
        let before = cycles;
        cycles += setup_time;
        setup_cycles += setup_time;
        setup_bytes += bytes;
        let kernel_start = cycles;
        let workload: Workload = serde_json::from_value(part["workload"].clone()).unwrap();
        let mut kernel = Sim::new(workload, cfg.clone());
        kernel.time_origin = kernel_start;
        let report = kernel.run();
        assert_eq!(report["drained"], true);
        assert_eq!(report["ownership_k_order_capacity_checks"], true);
        assert_eq!(u(&report,"dma_transactions_accepted"),u(&report,"dma_transactions_landed"));
        for field in sum_fields { add_tree(&mut sums[field], &report[field]); }
        for field in max_fields { sums[field] = json!(sums[field].as_u64().unwrap_or(0).max(u(&report, field))); }
        add_tree(&mut sums["prefetch_gate_wait_core_cycles"], &report["prefetch_gate_wait_core_cycles"]);
        for c in 0..nc {
            let target = &mut total_core[c];
            let child = &report["cores"][c];
            merge_stats(&mut target["stats"], &child["stats"], kernel_start);
            for field in ["weight_bank_words", "x_bank_words", "workspace_bank_words"] { add_tree(&mut target[field], &child[field]); }
            target["reserved_input_result_control"] = json!(target["reserved_input_result_control"].as_u64().unwrap().max(u(child,"reserved_input_result_control")));
            target["stats"]["workspace_peak_bytes"] = json!(target["stats"]["workspace_peak_bytes"].as_u64().unwrap_or(0)
                .max(u(child,"reserved_input_result_control")));
            add_tree(&mut target["stats"]["front_states"]["outer_chunk_setup"], &json!(setup_time));
            if c == owner {
                add_tree(&mut target["stats"]["control_cycles"], &json!(control_time));
                add_tree(&mut target["workspace_bank_words"], &json!(setup_words));
            }
        }
        if cfg.diagnostic_profile {
            let p = &report["m0_profile"];
            assert_eq!(p["mutually_exclusive"], true);
            for key in ["hbm_states", "core_states", "issue_interval_histograms", "stage_totals_count_operand_mac_rmw"] { add_tree(&mut profile[key], &p[key]); }
            let gated = (before..kernel_start).filter(|t| *t < cfg.dma_ready_after
                || *t % cfg.dma_ready_period >= cfg.dma_ready_cycles).count() as u64;
            add_tree(&mut profile["hbm_states"]["H3"], &json!(setup_time-gated));
            add_tree(&mut profile["hbm_states"]["H4"], &json!(gated));
            for c in 0..nc { add_tree(&mut profile["core_states"][c][if c==owner {"C8"} else {"C0"}], &json!(setup_time)); }
            for sample in p["issue_stage_samples"].as_array().unwrap() {
                let mut sample=sample.clone(); sample["outer_chunk"] = json!(index); sample["kernel_start_cycle"] = json!(kernel_start);
                profile["issue_stage_samples"].as_array_mut().unwrap().push(sample);
            }
        }
        for event in report["dispatch_audit"].as_array().unwrap() {
            let mut event=event.clone(); event["outer_chunk"] = json!(index); event["kernel_start_cycle"] = json!(kernel_start); audits.push(event);
        }
        for event in report["trace"].as_array().unwrap() {
            let mut event=event.clone(); event["outer_chunk"] = json!(index); event["kernel_start_cycle"] = json!(kernel_start); traces.push(event);
        }
        cycles += u(&report,"cycles");
        if out.is_null() { out = report.clone(); }
        parts.push(json!({"token_range":part["token_range"],"start_cycle":before,"setup_cycles":setup_time,
            "setup_control_cycles":control_time,"setup_sram_bytes":bytes,"kernel_start_cycle":kernel_start,"kernel_report":report}));
    }
    assert_eq!(u(&sums,"useful_macs"),u(plan,"expected_useful_macs"));
    assert_eq!(u(&sums,"weight_bytes"),u(plan,"weight_read_bytes"));
    assert!(u(&sums,"weight_bytes") >= u(plan,"unique_weight_bytes"));
    for (field, value) in sums.as_object().unwrap() { out[field] = value.clone(); }
    out["cycles"] = json!(cycles);
    out["time_ms_at_1ghz"] = json!(cycles as f64 / 1e6);
    out["experts_done_cycles"] = json!(parts.last().unwrap()["kernel_start_cycle"].as_u64().unwrap() + u(&parts.last().unwrap()["kernel_report"],"experts_done_cycles"));
    out["total_chunk_combine_cycles"] = out["combine_tail_cycles"].clone();
    out["combine_tail_cycles"] = parts.last().unwrap()["kernel_report"]["combine_tail_cycles"].clone();
    out["workload"] = parent["id"].clone();
    out["cores"] = json!(total_core);
    out["dispatch_audit"] = json!(audits);
    out["trace"] = json!(traces);
    out["unique_weight_bytes"] = plan["unique_weight_bytes"].clone();
    out["refetch_bytes"] = json!(u(&out,"weight_bytes") - u(plan,"unique_weight_bytes"));
    out["physical_budget"] = parent["engine_layout"]["hardware"].clone();
    out["control_accounting"] = plan["chunks"][0]["workload"]["engine_layout"]["control_accounting"].clone();
    // Per-chunk adaptive states/diagnostics are kept verbatim below. There is
    // deliberately no claim that resetting them equals one infeasible run.
    for field in ["supply_diagnostics", "joint_diagnostics", "feedback_q8"] { out.as_object_mut().unwrap().remove(field); }
    out["scope"] = json!("Previously infeasible legacy large-window capacity extension: serial unchanged joint_v1 kernels, global private X/Y/routes retained, repeated HBM weights charged; not native Ramulator or full-model inference");
    out["legacy_chunks"] = json!({"schema":"plena_legacy_token_chunks_v1","chunk_size":plan["chunk_size"],
        "chunks":parts.len(),"full_layer_persistent_cores":plan["persistent_cores"],"control_record":plan["control_record"],
        "setup_cycles":setup_cycles,"setup_sram_bytes":setup_bytes,"refetch_bytes":out["refetch_bytes"],
        "reset_policy":"all kernel DMA/contexts/ports drain; old adaptive state and weight streams restart per chunk; physical DMA-ready pattern uses the global cycle; no cross-chunk cache or overlap",
        "setup_policy":"read real resident token/expert descriptors using existing bank ports, serial charged route sequencer, write 32B control record; no X/Y copies due to row aliases",
        "parts":parts,"global_routes_and_scores_preserved":true,"persistent_addresses_checked":true});
    if cfg.diagnostic_profile {
        let hsum:u64 = profile["hbm_states"].as_object().unwrap().values().map(|x|x.as_u64().unwrap()).sum();
        let csums:Vec<u64> = profile["core_states"].as_array().unwrap().iter().map(|x|x.as_object().unwrap().values().map(|x|x.as_u64().unwrap()).sum()).collect();
        assert_eq!(hsum,cycles);assert!(csums.iter().all(|x|*x==cycles));
        profile["hbm_sum"] = json!(hsum);profile["core_sums"] = json!(csums);profile["mutually_exclusive"] = json!(true);
        profile["scope"] = json!("exclusive concatenation of unchanged chunk kernels; setup is H3 (H4 only if actual DMA-ready=false), owner C8/other cores C0; issue intervals exclude full-drain chunk boundaries; samples retain local times with explicit kernel_start_cycle");
        out["m0_profile"] = profile;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn additive_counters_and_peak_extrema_are_distinct() {
        let mut dst=json!({});
        merge_stats(&mut dst,&json!({"issues":3,"weight_peak_bytes":90,"eligible_wait_max_cycles":14,"done_cycle":11,"front_states":{"issue":3}}),20);
        merge_stats(&mut dst,&json!({"issues":5,"weight_peak_bytes":64,"eligible_wait_max_cycles":9,"done_cycle":7,"front_states":{"issue":5}}),50);
        assert_eq!(dst,json!({"issues":8,"weight_peak_bytes":90,"eligible_wait_max_cycles":14,"done_cycle":57,"front_states":{"issue":8}}));
    }
    #[test]
    fn physical_dma_ready_pattern_keeps_the_global_clock() {
        let w=Workload{id:"clock-origin".into(),batch:1,hidden:512,top_k:1,engine_layout:Value::Null,
            experts:vec![Expert{id:0,is_shared:false,m:1,h:512,f:16,token_indices:vec![0],weights:Value::Null}]};
        let mut cfg=Config::default();cfg.dma_ready_after=7;cfg.dma_ready_period=4;cfg.dma_ready_cycles=2;
        let mut s=Sim::new(w,cfg);s.time_origin=6;
        assert!(!s.dma_ready_now());s.now=1;assert!(!s.dma_ready_now());
        s.now=2;assert!(s.dma_ready_now());s.now=3;assert!(s.dma_ready_now());
        s.now=4;assert!(!s.dma_ready_now());
    }
}
