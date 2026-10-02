//! Causal integration checks, including cases absent from the large shape study.
use super::*;
fn input() -> Value {
    json!({"workload":{"id":"causal","batch":10,"hidden":544,"top_k":1,"experts":[{"id":1,"Me":2,"H":544,"F":32,"is_shared":false,"token_indices":[0,9]}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P0","comp_mode":"none","rank_lanes":0,"record_trace":true,"trace_limit":20000,"t_chunk":96,"dot_latency":20}})
}
#[test]
fn nonadjacent_routed_rows_use_actual_gather_strides() {
    let r = run(&input()).unwrap();
    let loads: Vec<_> = r["trace"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["event"] == "x_load" && v["source"] == 0 && v["k_segment"] == 0)
        .collect();
    assert!(!loads.is_empty());
    let spans = loads[0]["spans"].as_array().unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0][0], 0);
    assert_eq!(spans[1][0], 9 * 544 * 2);
    assert_eq!(spans[0][1], 512 * 2);
    assert_eq!(spans[1][1], 512 * 2);
    assert_eq!(r["invariants"]["gather_addresses_validated"], true);
}
#[test]
fn frozen_physical_capacity_does_not_shrink_with_batch() {
    let a = run(&input()).unwrap();
    let mut q = input();
    q["workload"]["batch"] = json!(16);
    let b = run(&q).unwrap();
    assert_eq!(a["budget"]["total_bytes"], b["budget"]["total_bytes"]);
    assert_eq!(a["budget"]["structures"], b["budget"]["structures"]);
    assert_eq!(a["budget"]["t_chunk_capacity"], 96);
}
#[test]
fn physical_mac_latency_remains_with_ideal_supply() {
    let mut a = input();
    a["config"]["ideal_hbm"] = json!(true);
    a["config"]["ideal_onchip"] = json!(true);
    a["config"]["dot_latency"] = json!(1);
    let fast = run(&a).unwrap();
    a["config"]["dot_latency"] = json!(20);
    let real = run(&a).unwrap();
    assert!(real["cycles"].as_u64().unwrap() > fast["cycles"].as_u64().unwrap());
    let issues: Vec<_> = real["trace"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["event"] == "issue")
        .collect();
    assert!(!issues.is_empty());
    for e in issues {
        assert!(e["accumulator_complete"].as_u64().unwrap() >= e["cycle"].as_u64().unwrap() + 20);
    }
}
#[test]
fn bounded_wor_and_ordered_feedback_all_complete() {
    let r = run(&input()).unwrap();
    assert_eq!(r["invariants"]["k_order_preserved"], true);
    assert!(
        r["actual_storage_peak_bytes"].as_u64().unwrap()
            <= r["budget"]["capacity_bytes"].as_u64().unwrap()
    );
    for core in r["cores"].as_array().unwrap() {
        assert!(
            core["wor_peak_slots"].as_u64().unwrap()
                <= core["wor_capacity_slots"].as_u64().unwrap()
        );
        assert_eq!(core["state_sum"], r["cycles"]);
    }
    assert_eq!(r["dma_transactions_accepted"], r["dma_transactions_landed"]);
}
#[test]
fn equal_payload_compensation_is_real_dma_not_posthoc_time() {
    let mut q = input();
    q["config"]["precision"] = json!("P1");
    q["config"]["main_bits"] = json!(4);
    q["config"]["rank_lanes"] = json!(8);
    q["config"]["ranks"] = json!({"routed":[8,8,8],"shared":[8,8,8]});
    q["config"]["comp_equal_bytes"] = json!(true);
    q["config"]["comp_mode"] = json!("none");
    let none = run(&q).unwrap();
    q["config"]["comp_mode"] = json!("lanes");
    let lane = run(&q).unwrap();
    assert_eq!(none["weight_bytes"], lane["weight_bytes"]);
    assert_eq!(
        none["dma_transactions_accepted"],
        lane["dma_transactions_accepted"]
    );
    assert!(none["comp_padding_bytes"].as_u64().unwrap() > 0);
    assert_eq!(none["pool_write_bytes"], none["weight_bytes"]);
}
#[test]
fn mxint8_split_prepass_counts_both_passes_and_drains() {
    let mut q = input();
    q["config"]["precision"] = json!("P2");
    q["config"]["factor_a"] = json!("mxint8s");
    q["config"]["factor_b"] = json!("bf16");
    q["config"]["rank_lanes"] = json!(8);
    q["config"]["comp_mode"] = json!("lanes");
    q["config"]["ranks"] = json!({"routed":[8,8,8],"shared":[8,8,8]});
    q["config"]["max_cycles"] = json!(100000);
    let two = run(&q).unwrap();
    q["config"]["factor_a"] = json!("mxint4");
    let one = run(&q).unwrap();
    let n = |r: &Value| {
        r["cores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["prepass_issues"].as_u64().unwrap())
            .sum::<u64>()
    };
    assert_eq!(n(&two), 2 * n(&one));
    assert_eq!(two["drained"], true);
}
