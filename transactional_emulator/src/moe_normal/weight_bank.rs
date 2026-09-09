//! Immutable, route-independent encoded weight bank binding for workload V2.
//! Identity checks happen before any simulated request. They are setup work,
//! outside the timed operator, and do not imply a persistent on-chip cache.
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{Expert, Workload};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bank {
    schema_version: u32,
    kind: String,
    input_dim: usize,
    expert_hidden_dim: usize,
    experts: Vec<Expert>,
    weight_format: String,
    weight_layout: String,
    block_size: usize,
    scale_axis: String,
    hbm_file: String,
    hbm_bytes: u64,
    hbm_sha256: String,
    /// Provenance is not used to regenerate or execute weights.
    generator: Generator,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generator {
    kind: String,
    seed: u64,
    source_sha256: String,
}

#[derive(Debug)]
pub struct BankBinding {
    pub manifest_path: PathBuf,
    pub image_path: PathBuf,
    pub image_bytes: u64,
    pub image_sha256: String,
    pub manifest_sha256: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn check_format(value: &Value, required: bool) -> Result<(), String> {
    for (key, expected) in [
        ("weight_format", json!("plena_e4m3_e8m0_block8")),
        ("weight_layout", json!("output_major_N_K")),
        ("block_size", json!(8)),
        ("scale_axis", json!("per_row_reduction_K")),
    ] {
        match value.get(key) {
            Some(actual) if *actual != expected => {
                return Err(format!("unsupported {key}: {actual}"));
            }
            None if required => return Err(format!("missing weight format field {key}")),
            _ => {}
        }
    }
    Ok(())
}

pub fn validate_contract(w: &Workload) -> Result<(), String> {
    match (w.schema_version, &w.weight_bank) {
        (1, None) => {}
        (2, Some(reference))
            if !reference.manifest.is_empty() && valid_digest(&reference.sha256) => {}
        _ => {
            return Err(
                "workload V1 must omit weight_bank; V2 requires manifest and SHA256".into(),
            );
        }
    }
    let metadata = w.metadata.as_ref().unwrap_or(&Value::Null);
    check_format(metadata, w.schema_version == 2)?;
    if w.schema_version == 2 {
        let reference = w.weight_bank.as_ref().unwrap();
        if metadata["bank_sha256"].as_str() != Some(reference.sha256.as_str()) {
            return Err("workload metadata bank identity differs from weight_bank".into());
        }
        if !metadata["hbm_sha256"].as_str().is_some_and(valid_digest) {
            return Err("workload V2 requires encoded image SHA256".into());
        }
    }
    Ok(())
}

fn parse_binding(
    w: &Workload,
    bytes: &[u8],
    manifest_path: PathBuf,
) -> Result<BankBinding, String> {
    validate_contract(w)?;
    let reference = w.weight_bank.as_ref().ok_or("no weight bank reference")?;
    let digest = format!("{:x}", Sha256::digest(bytes));
    if digest != reference.sha256 {
        return Err("weight bank manifest SHA256 mismatch".into());
    }
    let bank: Bank =
        serde_json::from_slice(bytes).map_err(|e| format!("invalid weight bank: {e}"))?;
    if bank.schema_version != 1 || bank.kind != "plena_moe_weight_bank" {
        return Err("unsupported weight bank schema/kind".into());
    }
    if bank.generator.kind != "synthetic_full_shape_v1"
        || !valid_digest(&bank.generator.source_sha256)
    {
        return Err("unsupported bank generator metadata".into());
    }
    // The seed is provenance, never a source of runtime-generated weights.
    let _seed = bank.generator.seed;
    check_format(
        &json!({
            "weight_format": bank.weight_format, "weight_layout": bank.weight_layout,
            "block_size": bank.block_size, "scale_axis": bank.scale_axis,
        }),
        true,
    )?;
    if bank.input_dim != w.input_dim
        || bank.expert_hidden_dim != w.expert_hidden_dim
        || serde_json::to_value(&bank.experts).unwrap() != serde_json::to_value(&w.experts).unwrap()
    {
        return Err("window dimensions/full expert catalog differ from immutable bank".into());
    }
    if bank.hbm_file != "weights.bin"
        || bank.hbm_bytes == 0
        || !bank.hbm_bytes.is_multiple_of(64)
        || !valid_digest(&bank.hbm_sha256)
        || w.metadata.as_ref().unwrap()["hbm_sha256"].as_str() != Some(bank.hbm_sha256.as_str())
    {
        return Err("invalid bank image length/path/identity".into());
    }
    // Canonical banks do not alias any element/scale row. Include padding in
    // the ownership interval so a tail cannot silently borrow another tensor.
    if bank.input_dim == 0 || bank.expert_hidden_dim == 0 || bank.experts.is_empty() {
        return Err("bank requires positive dimensions and a nonempty expert catalog".into());
    }
    if bank.experts.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return Err("bank expert IDs must be unique and sorted".into());
    }
    let mut ranges = Vec::new();
    for expert in &bank.experts {
        for (region, rows, cols) in [
            (&expert.gate, bank.expert_hidden_dim, bank.input_dim),
            (&expert.up, bank.expert_hidden_dim, bank.input_dim),
            (&expert.down, bank.input_dim, bank.expert_hidden_dim),
        ] {
            if region.rows != rows || region.cols != cols {
                return Err("bank matrix shape differs from common expert dimensions".into());
            }
            let scales = region.cols.div_ceil(8) as u64;
            let elements = scales.checked_mul(8).ok_or("bank row width overflow")?;
            for (base, stride, width) in [
                (region.element_base, region.element_row_stride, elements),
                (region.scale_base, region.scale_row_stride, scales),
            ] {
                if stride != width || !base.is_multiple_of(64) {
                    return Err("invalid bank stride/alignment".into());
                }
                let end = (region.rows as u64)
                    .checked_mul(stride)
                    .and_then(|n| n.checked_add(base))
                    .ok_or("bank address overflow")?;
                if end > bank.hbm_bytes {
                    return Err("bank region exceeds image".into());
                }
                ranges.push((base, end));
            }
        }
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|p| p[0].1 > p[1].0) {
        return Err("bank payload regions overlap".into());
    }
    let image_path = manifest_path
        .parent()
        .ok_or("bank manifest has no parent")?
        .join(bank.hbm_file);
    Ok(BankBinding {
        manifest_path,
        image_path,
        image_bytes: bank.hbm_bytes,
        image_sha256: bank.hbm_sha256,
        manifest_sha256: digest,
    })
}

pub fn resolve(w: &Workload, workload_path: &Path) -> Result<Option<BankBinding>, String> {
    validate_contract(w)?;
    let Some(reference) = &w.weight_bank else {
        return Ok(None);
    };
    let path = workload_path
        .parent()
        .ok_or("workload has no parent")?
        .join(&reference.manifest)
        .canonicalize()
        .map_err(|e| format!("weight bank manifest: {e}"))?;
    let bytes = std::fs::read(&path).map_err(|e| format!("weight bank manifest: {e}"))?;
    let mut binding = parse_binding(w, &bytes, path)?;
    binding.image_path = binding
        .image_path
        .canonicalize()
        .map_err(|e| format!("weight bank image: {e}"))?;
    let window_image = workload_path
        .parent()
        .unwrap()
        .join(&w.hbm_file)
        .canonicalize()
        .map_err(|e| format!("window image: {e}"))?;
    if binding.image_path != window_image {
        return Err("window must reference the immutable bank image file".into());
    }
    Ok(Some(binding))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Workload, Value) {
        let region = |e, s| {
            json!({"rows":1,"cols":1,"element_base":e,"scale_base":s,
            "element_row_stride":8,"scale_row_stride":1})
        };
        let bank = json!({"schema_version":1,"kind":"plena_moe_weight_bank",
            "input_dim":1,"expert_hidden_dim":1,"experts":[{"id":7,
                "gate":region(0,64),"up":region(128,192),"down":region(256,320)}],
            "weight_format":"plena_e4m3_e8m0_block8","weight_layout":"output_major_N_K",
            "block_size":8,"scale_axis":"per_row_reduction_K","hbm_file":"weights.bin",
            "hbm_bytes":384,"hbm_sha256":"a".repeat(64),
            "generator":{"kind":"synthetic_full_shape_v1","seed":0,"source_sha256":"c".repeat(64)}});
        let workload: Workload = serde_json::from_value(json!({
            "schema_version":2,"name":"bank_test","hbm_file":"weights.bin",
            "input_dim":1,"expert_hidden_dim":1,"inputs_bf16":[[0]],
            "routes":[{"token":0,"slot":0,"expert":7,"weight":1.0}],
            "experts":bank["experts"],"weight_bank":{"manifest":"bank.json","sha256":"b".repeat(64)},
            "metadata":{"weight_format":bank["weight_format"],"weight_layout":bank["weight_layout"],
                "block_size":8,"scale_axis":bank["scale_axis"],"hbm_sha256":bank["hbm_sha256"],"bank_sha256":"b".repeat(64)}
        })).unwrap();
        (workload, bank)
    }

    fn bind(w: &mut Workload, bank: &Value) -> Vec<u8> {
        let bytes = serde_json::to_vec(bank).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        w.weight_bank.as_mut().unwrap().sha256 = digest.clone();
        w.metadata.as_mut().unwrap()["bank_sha256"] = json!(digest);
        bytes
    }

    #[test]
    fn verifies_identity_and_rejects_repacking_or_unsupported_scales() {
        let (mut w, bank) = fixture();
        let bytes = bind(&mut w, &bank);
        assert!(parse_binding(&w, &bytes, "/bank/bank.json".into()).is_ok());
        let mut changed = bytes.clone();
        changed.push(b' ');
        assert!(
            parse_binding(&w, &changed, "/bank/bank.json".into())
                .unwrap_err()
                .contains("SHA256")
        );
        w.experts[0].gate.element_base += 64;
        assert!(parse_binding(&w, &bytes, "/bank/bank.json".into()).is_err());
        w.experts[0].gate.element_base -= 64;
        w.metadata.as_mut().unwrap()["block_size"] = json!(32);
        assert!(validate_contract(&w).is_err());
    }

    #[test]
    fn rejects_aliasing_even_when_catalog_and_identity_agree() {
        let (mut w, mut bank) = fixture();
        bank["experts"][0]["up"]["element_base"] = json!(0);
        w.experts[0].up.element_base = 0;
        let bytes = bind(&mut w, &bank);
        let error = parse_binding(&w, &bytes, "/bank/bank.json".into())
            .err()
            .unwrap();
        assert!(error.contains("overlap"));
    }

    #[test]
    fn rejects_noncanonical_catalogs_even_when_window_hashes_are_rebound() {
        let mutations: [fn(&mut Value); 7] = [
            |b| b["hbm_file"] = json!("other.bin"),
            |b| b["generator"]["kind"] = json!("unknown"),
            |b| b["generator"]["source_sha256"] = json!("bad digest"),
            |b| b["experts"][0]["gate"]["element_row_stride"] = json!(16),
            |b| b["experts"][0]["down"]["rows"] = json!(2),
            |b| {
                let entry = b["experts"][0].clone();
                b["experts"].as_array_mut().unwrap().push(entry);
            },
            |b| b["experts"] = json!([]),
        ];
        for mutate in mutations {
            let (mut w, mut bank) = fixture();
            mutate(&mut bank);
            w.experts = serde_json::from_value(bank["experts"].clone()).unwrap();
            let bytes = bind(&mut w, &bank);
            assert!(parse_binding(&w, &bytes, "/bank/bank.json".into()).is_err());
        }
    }
}
