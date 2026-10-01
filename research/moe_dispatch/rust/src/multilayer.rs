//! Sequential multi-layer FFN baseline. No weight or credit state crosses a
//! layer boundary: this measures the opportunity for future cross-layer work,
//! without granting any prefetch benefit to the baseline.
use super::*;

#[derive(Deserialize)]
struct Layer {
    workload: Workload,
    #[serde(default)]
    gap_after_cycles: u64,
    #[serde(default)]
    gap_hbm_bytes: u64,
}

#[derive(Deserialize)]
struct Input {
    schema: String,
    config: Config,
    layers: Vec<Layer>,
    #[serde(default = "default_tail_cycles")]
    opportunity_tail_cycles: u64,
}

fn default_tail_cycles() -> u64 {
    65536
}

pub(super) fn run(value: Value) -> Value {
    let mut input: Input =
        serde_json::from_value(value).expect("invalid multilayer baseline input");
    assert_eq!(input.schema, "plena_moe_multilayer_baseline_input_v1");
    assert!(
        input.layers.len() >= 2,
        "a baseline requires at least two layers"
    );
    assert!(input.opportunity_tail_cycles > 0);
    assert!(input.config.runtime_fsm && input.config.split == "none");
    assert!(!input.config.ideal_hbm && !input.config.ideal_onchip);
    assert!(
        !input.config.record_trace,
        "use bounded profile bins, not a full per-request trace"
    );
    if input.config.profile_bin_cycles == 0 {
        input.config.profile_bin_cycles = 1024;
    }
    let mut now = 0u64;
    let mut weight_bytes = 0u64;
    let mut gap_hbm_bytes = 0u64;
    let mut rows = Vec::with_capacity(input.layers.len());
    for (index, layer) in input.layers.into_iter().enumerate() {
        assert!(
            layer.gap_hbm_bytes
                <= layer
                    .gap_after_cycles
                    .saturating_mul(input.config.hbm_bytes_per_ns as u64),
            "external gap traffic exceeds configured HBM byte bandwidth"
        );
        let start = now;
        let mut report = Sim::new(layer.workload, input.config.clone()).run();
        let cycles = report["cycles"].as_u64().unwrap();
        let bytes = report["weight_bytes"].as_u64().unwrap();
        weight_bytes += bytes;
        let tail_start = cycles.saturating_sub(input.opportunity_tail_cycles);
        let mut opportunity = vec![0u64; input.config.lanes.len()];
        let mut spare_hbm_bytes = 0u64;
        for bin in report["profile_bins"].as_array().unwrap() {
            if bin["start_cycle"].as_u64().unwrap() + bin["cycles"].as_u64().unwrap() <= tail_start
            {
                continue;
            }
            // Bin boundaries can include a partial tail; the full bins remain
            // available below for exact downstream time-series analysis.
            spare_hbm_bytes += bin["hbm_spare_bytes_sum"].as_u64().unwrap();
            for (c, v) in opportunity.iter_mut().enumerate() {
                *v += bin["slot_credit_bw_opportunity_cycles"][c]
                    .as_u64()
                    .unwrap();
            }
        }
        if let Some(bins) = report["profile_bins"].as_array_mut() {
            for bin in bins {
                bin["global_start_cycle"] = json!(start + bin["start_cycle"].as_u64().unwrap());
            }
        }
        now += cycles;
        let end = now;
        let gap_end = end + layer.gap_after_cycles;
        gap_hbm_bytes += layer.gap_hbm_bytes;
        rows.push(json!({"index":index,"workload_id":report["workload"],
            "ffn_start_cycle":start,"ffn_end_cycle":end,"ffn_cycles":cycles,
            "gap_start_cycle":end,"gap_end_cycle":gap_end,
            "gap_cycles_external":layer.gap_after_cycles,
            "gap_hbm_bytes_external":layer.gap_hbm_bytes,
            "tail_profile_requested_cycles":input.opportunity_tail_cycles,
            "tail_bin_spare_hbm_bytes_upper":spare_hbm_bytes,
            "tail_slot_credit_bw_opportunity_core_cycles_upper":opportunity,
            "report":report}));
        now = gap_end;
    }
    json!({"schema":"plena_moe_multilayer_baseline_report_v1",
        "scope":"sequential FFN layers with caller-supplied intervening time/traffic; no cross-layer prefetch, no simulated Attention or Router execution",
        "no_cross_layer_prefetch":true,
        "fixed_config":input.config,
        "layers":rows,"total_cycles":now,"weight_bytes":weight_bytes,
        "external_gap_hbm_bytes":gap_hbm_bytes,
        "configured_hbm_byte_utilization":(weight_bytes + gap_hbm_bytes) as f64 / (now as f64 * input.config.hbm_bytes_per_ns as f64)})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_layers_have_separate_credits_and_absolute_profile_time() {
        let workload = Workload {
            id: "layer".into(),
            batch: 2,
            hidden: 512,
            top_k: 1,
            engine_layout: Value::Null,
            experts: vec![Expert {
                id: 0,
                is_shared: false,
                m: 2,
                h: 512,
                f: 128,
                token_indices: vec![0, 1],
                weights: Value::Null,
            }],
        };
        let cfg = Config {
            lanes: vec![6],
            profile_bin_cycles: 64,
            ..Default::default()
        };
        let value = json!({"schema":"plena_moe_multilayer_baseline_input_v1",
            "config":cfg,"layers":[
                {"workload":workload,"gap_after_cycles":100,"gap_hbm_bytes":3200},
                {"workload":workload,"gap_after_cycles":0,"gap_hbm_bytes":0}]});
        let result = run(value);
        let layers = result["layers"].as_array().unwrap();
        assert_eq!(result["no_cross_layer_prefetch"], true);
        assert_eq!(layers[1]["ffn_start_cycle"], layers[0]["gap_end_cycle"]);
        assert_eq!(
            layers[0]["report"]["weight_bytes"],
            layers[1]["report"]["weight_bytes"]
        );
        assert_eq!(
            layers[1]["report"]["credit_peak"],
            layers[0]["report"]["credit_peak"]
        );
        let bins = layers[1]["report"]["profile_bins"].as_array().unwrap();
        assert_eq!(bins[0]["global_start_cycle"], layers[1]["ffn_start_cycle"]);
        assert_eq!(
            bins.iter()
                .map(|b| b["cycles"].as_u64().unwrap())
                .sum::<u64>(),
            layers[1]["ffn_cycles"].as_u64().unwrap()
        );
        assert_eq!(
            bins.iter()
                .map(|b| b["hbm_accepted_bytes"].as_u64().unwrap())
                .sum::<u64>(),
            layers[1]["report"]["weight_bytes"].as_u64().unwrap()
        );
    }
}
