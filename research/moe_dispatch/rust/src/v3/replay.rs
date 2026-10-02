//! BF16 storage and segmented FP32 arithmetic reference replay.
use serde_json::{Value, json};
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn mat(v: &Value) -> Result<Vec<Vec<f32>>, String> {
    v.as_array()
        .ok_or("matrix is not array")?
        .iter()
        .map(|r| {
            r.as_array()
                .ok_or("matrix row is not array")?
                .iter()
                .map(|x| {
                    x.as_f64()
                        .map(|z| z as f32)
                        .ok_or_else(|| "matrix value is not number".to_string())
                })
                .collect()
        })
        .collect()
}
fn proj(
    x: &[Vec<f32>],
    w: &[Vec<f32>],
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    placement: &Value,
    l: usize,
    streamed: bool,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let m = x.len();
    let k = x.first().ok_or("empty X")?.len();
    let n = w.len();
    if x.iter().any(|r| r.len() != k) || w.iter().any(|r| r.len() != k) {
        return Err("W[N,K] shape mismatch".into());
    }
    let r = a.first().map_or(0, Vec::len);
    if r > 0
        && (a.len() != k
            || a.iter().any(|row| row.len() != r)
            || b.len() != r
            || b.iter().any(|z| z.len() != n))
    {
        return Err("A/B shape mismatch".into());
    }
    let mut u = vec![vec![0.0f32; r]; m];
    for row in 0..m {
        for j in 0..r {
            let mut sum = 0.0;
            for seg in (0..k).step_by(512) {
                let mut part = 0.0f32;
                for q in seg..(seg + 512).min(k) {
                    part += bf16(x[row][q]) * a[q][j];
                }
                sum += part;
            }
            u[row][j] = bf16(sum);
        }
    }
    let ns = k.div_ceil(512);
    let fused = r.min(if streamed { l } else { l * ns });
    if let Some(segments) = placement.as_array() {
        if segments.first().is_some_and(Value::is_array) {
            if segments.len() != ns {
                return Err("rank placement segment count mismatch".into());
            }
            let mut seen = vec![false; fused];
            for row in segments {
                let indices = row.as_array().ok_or("mixed rank placement format")?;
                if indices.len() > l {
                    return Err("rank placement exceeds installed rank lanes".into());
                }
                for j in indices {
                    let j = j.as_u64().ok_or("invalid rank index")? as usize;
                    if j >= fused || seen[j] {
                        return Err("rank placement duplicate or out of range".into());
                    }
                    seen[j] = true;
                }
            }
            if seen.iter().any(|&x| !x) {
                return Err("rank placement misses fused rank".into());
            }
        } else {
            if segments.len() < fused {
                return Err("rank placement too short".into());
            }
            let mut counts = vec![0; ns];
            for j in segments.iter().take(fused) {
                let i = j.as_u64().ok_or("invalid rank segment")? as usize;
                if i >= ns {
                    return Err("rank segment out of range".into());
                }
                counts[i] += 1;
            }
            if counts.iter().any(|&n| n > l) {
                return Err("rank segment exceeds installed lanes".into());
            }
        }
    }
    let segfor = |j: usize| -> usize {
        if j >= fused {
            return ns + (j - fused) / l.max(1);
        }
        if let Some(a) = placement.as_array() {
            if a.first().is_some_and(Value::is_array) {
                for (seg, rows) in a.iter().enumerate() {
                    if rows
                        .as_array()
                        .is_some_and(|rr| rr.iter().any(|v| v.as_u64() == Some(j as u64)))
                    {
                        return seg;
                    }
                }
            } else if let Some(v) = a.get(j).and_then(Value::as_u64) {
                return v as usize;
            }
        }
        if streamed { ns - 1 } else { j % ns }
    };
    let mut y = vec![vec![0.0; n]; m];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for seg in 0..ns {
                let mut part = 0.0;
                for q in seg * 512..((seg + 1) * 512).min(k) {
                    part += bf16(x[row][q]) * w[col][q];
                }
                let mut correction = 0.0;
                for j in 0..r {
                    if segfor(j) == seg {
                        correction += u[row][j] * b[j][col];
                    }
                }
                part += correction;
                sum += part;
            }
            for tail in (fused..r).step_by(l.max(1)) {
                let mut partial = 0.0;
                for j in tail..(tail + l).min(r) {
                    partial += u[row][j] * b[j][col];
                }
                sum += partial;
            }
            y[row][col] = sum;
        }
    }
    let _ = l;
    Ok((y, u))
}
pub fn replay(v: &Value) -> Result<Value, String> {
    let x = mat(&v["x"])?;
    let names = ["g", "u", "d"];
    let mut weights = vec![];
    let mut factors = vec![];
    for name in names {
        weights.push(mat(&v["weights"][name])?);
        let fv = &v["factors"][name];
        let a = if fv["a"].is_array() {
            mat(&fv["a"])?
        } else {
            vec![]
        };
        let b = if fv["b"].is_array() {
            mat(&fv["b"])?
        } else {
            vec![]
        };
        factors.push((a, b));
    }
    let l = v["L"].as_u64().unwrap_or(8) as usize;
    let (g, ug) = proj(
        &x,
        &weights[0],
        &factors[0].0,
        &factors[0].1,
        v.get("rank_segments")
            .or_else(|| v.get("state").and_then(|s| s.get("rank_segments")))
            .unwrap_or(&Value::Null)
            .get("g")
            .unwrap_or(&Value::Null),
        l,
        false,
    )?;
    let (u, uu) = proj(
        &x,
        &weights[1],
        &factors[1].0,
        &factors[1].1,
        v.get("rank_segments")
            .or_else(|| v.get("state").and_then(|s| s.get("rank_segments")))
            .unwrap_or(&Value::Null)
            .get("u")
            .unwrap_or(&Value::Null),
        l,
        false,
    )?;
    let gate = &v["gate"];
    let z = g
        .iter()
        .enumerate()
        .map(|(row, r)| {
            r.iter()
                .enumerate()
                .map(|(col, &q)| {
                    let scale = gate
                        .get(row)
                        .and_then(Value::as_f64)
                        .or_else(|| gate.as_f64())
                        .unwrap_or(1.0) as f32;
                    let sigmoid = if q >= 0.0 {
                        1.0 / (1.0 + (-q).exp())
                    } else {
                        let eq = q.exp();
                        eq / (1.0 + eq)
                    };
                    bf16((q * sigmoid * u[row][col]) * scale)
                })
                .collect()
        })
        .collect::<Vec<Vec<f32>>>();
    let (y, ud) = proj(
        &z,
        &weights[2],
        &factors[2].0,
        &factors[2].1,
        v.get("rank_segments")
            .or_else(|| v.get("state").and_then(|s| s.get("rank_segments")))
            .unwrap_or(&Value::Null)
            .get("d")
            .unwrap_or(&Value::Null),
        l,
        v["z_mode"] == "streamed",
    )?;
    let error = if v["output"].is_array() {
        let gold = mat(&v["output"])?;
        if gold.len() != y.len() || gold.iter().zip(&y).any(|(a, b)| a.len() != b.len()) {
            return Err("gold output shape mismatch".into());
        }
        let mut diff = 0.0f64;
        let mut norm = 0.0f64;
        for (row, a) in y.iter().enumerate() {
            for (col, &b) in a.iter().enumerate() {
                let gg = gold[row][col] as f64;
                diff += (b as f64 - gg).powi(2);
                norm += gg * gg;
            }
        }
        (diff / norm.max(1e-30)).sqrt()
    } else {
        0.0
    };
    Ok(
        json!({"schema":"plena_v3_numeric_replay_result_v1","hardware_relative_error":error,"hardware_pass":error<=1e-5,"output":y,"Z":z,"U_g":ug,"U_u":uu,"U_d":ud,"arithmetic":"X,U,Z BF16; ordered K-segment FP32 accumulation; compensation before SiLU; FP32 output"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_round_even() {
        assert_eq!(bf16(1.0), 1.0);
    }
    #[test]
    fn zero_compensator() {
        let v = json!({"x":[[1.0,2.0]],"weights":{"g":[[1.0,0.0]],"u":[[0.0,1.0]],"d":[[1.0]]},"factors":{},"gate":[1.0]});
        let out = replay(&v).unwrap();
        assert!((out["output"][0][0].as_f64().unwrap() - 1.4609375).abs() < 1e-6);
    }
}
