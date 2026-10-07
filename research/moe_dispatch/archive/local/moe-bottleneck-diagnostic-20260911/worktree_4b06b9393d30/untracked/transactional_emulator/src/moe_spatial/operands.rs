//! Explicit BF16 model operands. Never silently fall back to generated fixtures.
//! X is [M,K], W is [N,K], both row-major little-endian BF16.
use super::*;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorFile {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperandFile {
    pub expert: usize,
    pub x: TensorFile,
    pub w: TensorFile,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub jobs: Vec<OperandFile>,
}
pub struct Values {
    pub x: Vec<f32>,
    pub w: Vec<f32>,
}
pub struct Operands {
    pub jobs: Vec<Values>,
}

pub fn input_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    let m: Manifest = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut paths = vec![path.to_path_buf()];
    for j in m.jobs {
        paths.extend([base.join(j.x.path), base.join(j.w.path)]);
    }
    Ok(paths)
}

fn tensor(base: &Path, file: &TensorFile, count: usize) -> Result<Vec<f32>, String> {
    let path = base.join(&file.path);
    let size = std::fs::metadata(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if size != (count * 2) as u64 {
        return Err(format!(
            "{}: expected {} BF16 bytes, got {size}",
            path.display(),
            count * 2
        ));
    }
    let raw = std::fs::read(&path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&raw)) != file.sha256 {
        return Err(format!("{}: operand SHA256 mismatch", path.display()));
    }
    let values: Vec<_> = raw
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("external operands must be finite BF16".into());
    }
    Ok(values)
}
impl Operands {
    pub fn load(path: &Path, r: &Request) -> Result<Self, String> {
        validate(r)?;
        if !r.verify_values {
            return Err("external operands require verify_values=true".into());
        }
        let raw = std::fs::read(path).map_err(|e| e.to_string())?;
        let m: Manifest = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if m.schema != "plena_bf16_x_mk_w_nk_v1" || m.jobs.len() != r.jobs.len() {
            return Err("external operand schema/job count mismatch".into());
        }
        let base = path.parent().unwrap_or(Path::new("."));
        let mut jobs = Vec::new();
        for (f, j) in m.jobs.iter().zip(&r.jobs) {
            if f.expert != j.expert {
                return Err("external operand expert order mismatch".into());
            }
            jobs.push(Values {
                x: tensor(base, &f.x, j.m * j.k)?,
                w: tensor(base, &f.w, j.n * j.k)?,
            });
        }
        Ok(Self { jobs })
    }

    pub fn validate(&self, r: &Request) -> Result<(), String> {
        if !r.verify_values || self.jobs.len() != r.jobs.len() {
            return Err("external operand count/verification mismatch".into());
        }
        for (v, j) in self.jobs.iter().zip(&r.jobs) {
            if v.x.len() != j.m * j.k
                || v.w.len() != j.n * j.k
                || v.x
                    .iter()
                    .chain(&v.w)
                    .any(|x| !x.is_finite() || bf16::from_f32(*x).to_f32() != *x)
            {
                return Err("external operand dimensions or BF16 values invalid".into());
            }
        }
        Ok(())
    }

    /// Recursive reference, independent of tile loading, scheduling and the
    /// iterative datapath tree. K tiles accumulate in ascending order in FP32.
    pub fn reference(&self, r: &Request, index: usize) -> Vec<f32> {
        let j = &r.jobs[index];
        let v = &self.jobs[index];
        fn sum(x: &[f32], w: &[f32], start: usize, span: usize) -> f32 {
            if span == 1 {
                if start < x.len() {
                    x[start] * w[start]
                } else {
                    0.0
                }
            } else {
                sum(x, w, start, span / 2) + sum(x, w, start + span / 2, span / 2)
            }
        }
        let mut out = vec![0.0; j.m * j.n];
        for row in 0..j.m {
            for col in 0..j.n {
                let x = &v.x[row * j.k..(row + 1) * j.k];
                let w = &v.w[col * j.k..(col + 1) * j.k];
                for k in (0..j.k).step_by(r.k_lanes) {
                    out[row * j.n + col] += sum(x, w, k, r.k_lanes);
                }
            }
        }
        out
    }
}
